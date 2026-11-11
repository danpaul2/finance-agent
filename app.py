"""Streamlit UI (criterion 2).  Run:  streamlit run app.py

Tabs: Dashboard (bank-app style summary) | Chat (the agent) | Prompts (criterion 3) | Evaluation
"""
import traceback
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

import prompts as P
import rag
import tools as T
from agents.categorizer import categorize
from agents.parser import parse_csv, parse_statement
from config import get_llm
from router import FinanceAgent

st.set_page_config(page_title="Personal Finance Agent", page_icon="💸", layout="wide")
SAMPLE_CSV = Path(__file__).parent / "data" / "sample_statement.csv"
SAMPLE_PDF = Path(__file__).parent / "data" / "sample_statement.pdf"
SAMPLE = SAMPLE_CSV  # backwards-compatibility
SUGGESTIONS = ["Show me all my Rapido rides",
               "How much did I spend on food in September?",
               "Which category grew most between September and October?",
               "Did my balance ever go negative?",
               "Where am I overspending and how can I save more?"]


def inr(x: float) -> str:
    return f"{'-' if x < 0 else ''}\u20b9{abs(x):,.0f}"


# ------------------------------------------------------------------ pipeline
def process(source, label: str, password: str | None = None) -> None:
    tx, excluded = parse_statement(source, filename=label, password=password)
    if tx.empty:
        raise ValueError(
            "No valid transactions could be extracted from this statement. "
            "Please verify that the file contains transaction tables and is not an image-only scan."
        )
    tx["date"] = pd.to_datetime(tx["date"], errors="coerce")
    truth = tx.set_index("txn_id")["true_category"] if tx["true_category"].notna().any() else None
    bar = st.progress(0.0, text="Categorizing transactions...")
    df = categorize(tx.drop(columns=["true_category"]), progress=lambda p: bar.progress(p, text="Categorizing transactions..."))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bar.progress(1.0, text="Indexing in ChromaDB...")
    ctx = rag.RagContext.create(df)
    agent = FinanceAgent(df, get_llm(), ctx)
    bar.empty()
    st.session_state.update(df=df, excluded=excluded, truth=truth, agent=agent, label=label, messages=[])


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Statement")
    up = st.file_uploader("Upload a bank statement (CSV or PDF)", type=["csv", "pdf"], key="upload",
                          help="Upload a bank statement as CSV or PDF (e.g. HDFC, ICICI, SBI, Axis, or standard statements). CSV needs date, description and debit/credit or amount columns.")
    pdf_pwd = None
    if up is not None and up.name.lower().endswith(".pdf"):
        pdf_pwd = st.text_input("PDF Password (if encrypted)", type="password", key="pdf_pwd",
                                help="Leave blank if the statement is not password-protected.")
    use_sample = st.checkbox("Use the sample statement", value=True, key="use_sample", disabled=up is not None)
    sample_format = "CSV"
    if use_sample and up is None and SAMPLE_PDF.exists():
        sample_format = st.radio("Sample format", ["CSV", "PDF"], horizontal=True, key="sample_format")

    if st.button("Analyze", type="primary", key="analyze"):
        try:
            with st.spinner("Working..."):
                if up is not None:
                    process(up, up.name, password=pdf_pwd)
                elif use_sample:
                    if sample_format == "PDF" and SAMPLE_PDF.exists():
                        process(SAMPLE_PDF, "sample_statement.pdf (synthetic)")
                    else:
                        process(SAMPLE_CSV, "sample_statement.csv (synthetic)")
                else:
                    st.warning("Upload a statement (CSV or PDF) or tick the sample option.")
        except Exception as e:  # noqa: BLE001
            st.error(f"{type(e).__name__}: {str(e)[:600]}")
    if "df" in st.session_state:
        st.success(f"Loaded: {st.session_state.label}")
    st.caption("Use synthetic data for demos. Real statements contain personal information.")

st.title("Personal Finance Agent")
st.caption("Parser, Categorizer, Analysis and Advisor agents with intent routing and agentic RAG over ChromaDB.")

if "df" not in st.session_state:
    st.info("Choose a statement in the sidebar and press **Analyze**.")
    st.stop()

df, excluded, agent = st.session_state.df, st.session_state.excluded, st.session_state.agent
tab_dash, tab_chat, tab_prompts, tab_eval = st.tabs(["Dashboard", "Chat", "Prompts", "Evaluation"])

# ------------------------------------------------------------------ dashboard
with tab_dash:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = df["date"].dropna()
    if valid_dates.empty:
        st.warning("No valid transaction dates found in this statement.")
        st.stop()
    months = sorted(valid_dates.dt.strftime("%Y-%m").unique())
    month = st.selectbox("Period", ["All months"] + months, key="period")
    m = None if month == "All months" else month
    view = df if m is None else df[df["date"].dt.strftime("%Y-%m") == m]

    income = view[(view["type"] == "credit") & (view["category"] == "Income")]["amount"].sum()
    saved = -view[(view["type"] == "debit") & (view["category"] == "Savings Transfer")]["amount"].sum()
    c = st.columns(5)
    c[0].metric("Total spent", inr(T.total_spent(df, month=m)))
    c[1].metric("Income", inr(income))
    c[2].metric("Moved to savings", inr(saved))
    c[3].metric("Lowest balance", inr(view["balance"].min()))
    c[4].metric("Transactions", len(view))
    st.caption("Spending excludes savings transfers and transfers between accounts.")

    left, right = st.columns(2)
    cb = T.category_breakdown(df, m).reset_index()
    with left:
        st.subheader("Where the money went")
        if cb.empty:
            st.write("No spending in this period.")
        else:
            donut = alt.Chart(cb).mark_arc(innerRadius=70).encode(
                theta="total:Q", color=alt.Color("category:N", legend=alt.Legend(title=None)),
                tooltip=["category", alt.Tooltip("total:Q", format=",.0f"), "count", "pct"])
            st.altair_chart(donut, width="stretch")
    with right:
        st.subheader("Spending by category")
        if not cb.empty:
            bars = alt.Chart(cb).mark_bar().encode(
                x=alt.X("total:Q", title="INR"), y=alt.Y("category:N", sort="-x", title=None),
                tooltip=["category", alt.Tooltip("total:Q", format=",.0f"), "count", "pct"])
            st.altair_chart(bars, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Month by month")
        mt = T.monthly_trend(df).reset_index().rename(columns={"date": "month"})
        long = mt.melt(id_vars="month", value_vars=["spent", "income"], var_name="kind", value_name="INR")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("month:N", title=None), xOffset="kind:N", y="INR:Q", color=alt.Color("kind:N", title=None),
            tooltip=["month", "kind", alt.Tooltip("INR:Q", format=",.0f")]), width="stretch")
    with right:
        st.subheader("Top merchants")
        tm = T.top_merchants(df, 8).reset_index()
        st.altair_chart(alt.Chart(tm).mark_bar().encode(
            x=alt.X("total:Q", title="INR"), y=alt.Y("merchant:N", sort="-x", title=None),
            tooltip=["merchant", alt.Tooltip("total:Q", format=",.0f"), "count"]), width="stretch")

    od = T.overdraft_events(df)
    if len(od):
        st.warning(f"Balance below zero on {len(od)} transactions, first on {od['date'].min():%d %b %Y}; "
                   f"lowest {inr(od['balance'].min())}.")
    an = T.find_anomalies(df)
    if len(an):
        with st.expander(f"Unusually large debits ({len(an)})"):
            st.dataframe(an, hide_index=True, width="stretch")

    st.subheader("Transactions")
    f1, f2 = st.columns([2, 1])
    q = f1.text_input("Search description or merchant", key="txn_search")
    cats = f2.multiselect("Category", sorted(df["category"].unique()), key="txn_cats")
    t = view.copy()
    if q:
        t = t[t["text"].str.contains(q, case=False, na=False)]
    if cats:
        t = t[t["category"].isin(cats)]
    st.dataframe(t[["date", "description", "merchant", "category", "amount", "balance", "account", "cat_source"]],
                 hide_index=True, width="stretch",
                 column_config={"date": st.column_config.DateColumn("date", format="DD MMM YYYY"),
                                "amount": st.column_config.NumberColumn(format="%.2f"),
                                "balance": st.column_config.NumberColumn(format="%.2f")})
    if len(excluded):
        with st.expander(f"Excluded rows: {len(excluded)} failed or reversed transactions"):
            st.dataframe(excluded[["date", "description", "merchant", "status"]], hide_index=True, width="stretch")

# ------------------------------------------------------------------ chat
with tab_chat:
    st.write("Ask about your statement. Examples:")
    cols = st.columns(len(SUGGESTIONS))
    for i, s in enumerate(SUGGESTIONS):
        if cols[i].button(s, key=f"sugg{i}", width="stretch"):
            st.session_state["pending"] = s

    question = st.chat_input("Ask about your spending...", key="chat_box")
    question = st.session_state.pop("pending", None) or question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("The agent is working..."):
                r = agent.ask(question)
            st.session_state.messages.append({"role": "assistant", "content": r.answer, "intent": r.intent,
                                              "route": r.route_source, "trace": r.trace, "sources": r.sources})
        except Exception as e:
            import traceback

            st.error(f"{type(e).__name__}: {str(e)}")

            with st.expander("Debug traceback", expanded=True):
                st.code(traceback.format_exc())
                
            st.session_state.messages.append({
            "role": "assistant",
            "content": f"Something went wrong: {type(e).__name__}: {str(e)}",
            "trace": [],
            "sources": []
            })

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("trace"):
                with st.expander(f"How the agent answered (intent: {msg.get('intent')}, routed by {msg.get('route')})"):
                    for line in msg["trace"]:
                        st.markdown(f"- {line}")
                    for s in msg.get("sources", []):
                        st.caption(f"Retrieved guidance: {s['section']} ({s['source']})")

# ------------------------------------------------------------------ prompts
with tab_prompts:
    st.subheader("Prompt engineering")
    st.markdown("""
| Agent | Technique | LangChain / LCEL |
|---|---|---|
| Categorizer | Few-shot + short reasoning before the label | `ChatPromptTemplate \\| llm \\| PydanticOutputParser` |
| Intent router | Few-shot classification into a fixed schema | same pattern, then `RunnableBranch` |
| Analysis agent | **ReAct** (thought, action, observation) over pandas tools | structured step chain in a loop |
| Advisor | **Chain of thought** grounded in code-computed facts and retrieved guidance | structured chain |
| Reflection | Fact-check pass, triggered when a number is not supported | structured chain |
""")
    for title, system, user in [
        ("Categorizer", P.CATEGORIZER_SYSTEM, P.CATEGORIZER_USER),
        ("Intent router", P.ROUTER_SYSTEM, P.ROUTER_USER),
        ("Analysis agent (ReAct)", P.REACT_SYSTEM, P.REACT_USER),
        ("Transaction lookup", P.LOOKUP_SYSTEM, P.LOOKUP_USER),
        ("Advisor (CoT)", P.ADVISOR_SYSTEM, P.ADVISOR_USER),
        ("Reflection", P.REFLECT_SYSTEM, P.REFLECT_USER),
    ]:
        with st.expander(title):
            st.markdown("**System prompt**")
            st.code(system, language="text")
            st.markdown("**User prompt**")
            st.code(user.replace("\\n", "\n"), language="text")
    with st.expander("Few-shot example (Categorizer)"):
        st.code(P.FEWSHOT_INPUT + "\n\n" + P.FEWSHOT_OUTPUT, language="text")
    st.caption("`{format_instructions}` is filled in by `PydanticOutputParser` with the JSON schema of the expected output.")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.subheader("Categorizer evaluation")
    st.write("Label source counts:", df["cat_source"].value_counts().to_dict())
    truth = st.session_state.truth
    if truth is None:
        st.info("This file has no original category column to compare against.")
    else:
        cmp = df.set_index("txn_id")[["description", "merchant", "category"]].join(truth.rename("original"))
        ok = cmp["category"] == cmp["original"]
        st.metric("Agreement with the file's original labels", f"{ok.mean():.1%}", f"{int(ok.sum())} of {len(ok)} rows")
        shapes = cmp.groupby(["description", "merchant"]).apply(lambda g: (g["category"] == g["original"]).all(), include_groups=False)
        st.metric("Per unique transaction type", f"{shapes.mean():.1%}", f"{int(shapes.sum())} of {len(shapes)}")
        if (~ok).any():
            st.write("Disagreements:")
            st.dataframe(cmp[~ok], width="stretch")
        st.caption("Rule-based labels are included here. `python evaluate.py --no-rules` scores the LLM alone.")
