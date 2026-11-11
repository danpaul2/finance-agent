"""Analysis Agent tool layer: exact numbers (pandas/calculator) + retrieval, exposed to the ReAct loop.

The LLM only chooses WHICH tool to call with WHAT arguments; code computes every number.
"""
from __future__ import annotations
import inspect
import re
import pandas as pd
import tools as T

MAX_OBS_CHARS = 2500

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Spoken names that should hit the exact category strings in the statement.
_CATEGORY_ALIASES = {
    "food": "Food & Dining", "dining": "Food & Dining", "restaurant": "Food & Dining",
    "restaurants": "Food & Dining", "food and dining": "Food & Dining",
    "grocery": "Groceries", "groceries": "Groceries",
    "transport": "Transport", "travel": "Transport", "commute": "Transport",
    "shopping": "Shopping", "utilities": "Utilities", "bills": "Utilities",
    "subscription": "Subscriptions", "subscriptions": "Subscriptions",
    "entertainment": "Entertainment", "healthcare": "Healthcare", "health": "Healthcare",
    "medical": "Healthcare", "pharmacy": "Healthcare", "insurance": "Insurance",
    "education": "Education", "fitness": "Fitness", "gym": "Fitness",
    "housing": "Housing", "rent": "Housing", "household": "Household",
    "donations": "Donations", "donation": "Donations", "charity": "Donations",
    "cash": "Cash Withdrawal", "atm": "Cash Withdrawal", "cash withdrawal": "Cash Withdrawal",
    "bank charges": "Bank Charges", "fees": "Bank Charges", "charges": "Bank Charges",
    "miscellaneous": "Miscellaneous", "misc": "Miscellaneous",
    "income": "Income", "salary": "Income", "interest": "Interest", "cashback": "Cashback",
    "savings": "Savings Transfer", "savings transfer": "Savings Transfer",
    "transfer": "Transfers", "transfers": "Transfers",
}


def _available_months(df: pd.DataFrame) -> list[str]:
    dt = pd.to_datetime(df["date"], errors="coerce").dropna()
    return sorted(dt.dt.strftime("%Y-%m").unique())


def canon_month(df: pd.DataFrame, value) -> str | None:
    """Turn 'September', 'Sep 2026', or '2026-9' into a YYYY-MM that exists in the statement."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "all"}:
        return None
    available = _available_months(df)
    re_month = re.fullmatch(r"(\d{4})-(\d{1,2})", s)
    if re_month:
        ym = f"{re_month.group(1)}-{int(re_month.group(2)):02d}"
        return ym if ym in available else ym
    years = re.findall(r"20\d{2}", s)
    for token, num in _MONTH_NAMES.items():
        if re.search(rf"\b{token}\b", s.lower()):
            hits = [m for m in available if int(m[5:7]) == num]
            if years:
                hits = [h for h in hits if h[:4] in years] or hits
            if hits:
                return hits[-1]
    return s


def canon_category(df: pd.DataFrame, value) -> str | None:
    """Map 'food' or 'Food and Dining' onto the statement's category label."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "all"}:
        return None
    norm = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    labels = [str(c) for c in df["category"].dropna().unique()]
    for label in labels:
        if re.sub(r"[^a-z0-9]+", " ", label.lower()).strip() == norm:
            return label
    if norm in _CATEGORY_ALIASES:
        target = _CATEGORY_ALIASES[norm]
        for label in labels:
            if label.lower() == target.lower():
                return label
        return target
    # a unique category whose name contains the phrase ("dining" -> Food & Dining)
    hits = [label for label in labels if norm and norm in re.sub(r"[^a-z0-9]+", " ", label.lower())]
    if len(hits) == 1:
        return hits[0]
    return s


def _fmt(x) -> str:
    if isinstance(x, pd.DataFrame):
        return "(no rows)" if x.empty else x.to_string()
    return str(x)


def make_tools(df: pd.DataFrame, rag=None) -> dict:
    """name -> (callable, signature/description shown to the model)."""
    tools = {
        "total_spent": (lambda category=None, month=None: T.total_spent(
                            df, canon_category(df, category), canon_month(df, month)),
                        "total_spent(category?: str, month?: 'YYYY-MM') -> INR total of spending (excludes savings transfers)"),
        "category_breakdown": (lambda month=None: T.category_breakdown(df, canon_month(df, month)),
                               "category_breakdown(month?: 'YYYY-MM') -> total, count and % share per category"),
        "monthly_trend": (lambda: T.monthly_trend(df),
                          "monthly_trend() -> spending, income and month-over-month % change per month"),
        "category_mom": (lambda: T.category_mom(df),
                         "category_mom() -> per-category spend by month with change between the last two months"),
        "top_merchants": (lambda n=5: T.top_merchants(df, int(n)),
                          "top_merchants(n?: int) -> merchants with highest total spend"),
        "find_anomalies": (lambda: T.find_anomalies(df),
                           "find_anomalies() -> unusually large debits for their category"),
        "overdraft_events": (lambda: T.overdraft_events(df),
                             "overdraft_events() -> transactions where the account balance was negative"),
        "calculator": (lambda expr: T.calculator(str(expr)),
                       "calculator(expr: str) -> evaluates arithmetic such as '(8200-6100)/6100*100'"),
    }
    if rag is not None:
        tools["search_transactions"] = (
            lambda query, category=None, month=None: pd.DataFrame(
                rag.search_transactions(str(query), n=10, category=canon_category(df, category),
                                        month=canon_month(df, month)))[
                ["date", "merchant", "description", "category", "amount", "distance"]],
            "search_transactions(query: str, category?: str, month?: 'YYYY-MM') -> semantic search over transaction descriptions (ChromaDB)")
        tools["search_guidance"] = (
            lambda query: "\n".join(f"- [{g['section']}] {g['text']}" for g in rag.search_guidance(str(query), n=3)),
            "search_guidance(query: str) -> budgeting advice passages (ChromaDB)")
    return tools


def tool_descriptions(tools: dict) -> str:
    return "\n".join(f"- {desc}" for _, desc in tools.values())


def run_tool(tools: dict, name: str, args: dict | None) -> str:
    """Execute one tool call; errors come back as text so the agent can correct itself."""
    if name not in tools:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(tools)}"
    fn = tools[name][0]
    args = {k: v for k, v in (args or {}).items() if v not in (None, "", "null")}
    try:
        params = set(inspect.signature(fn).parameters)
        bad = set(args) - params
        if bad:
            return f"ERROR: unexpected argument(s) {sorted(bad)} for {name}. Expected: {sorted(params)}"
        out = _fmt(fn(**args))
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {e}"
    return out if len(out) <= MAX_OBS_CHARS else out[:MAX_OBS_CHARS] + "\n...(truncated)"


def data_info(df: pd.DataFrame) -> str:
    dt = pd.to_datetime(df["date"], errors="coerce")
    valid_dt = dt.dropna()
    months = sorted(valid_dt.dt.strftime("%Y-%m").unique())
    d_min = f"{valid_dt.min():%Y-%m-%d}" if not valid_dt.empty else "N/A"
    d_max = f"{valid_dt.max():%Y-%m-%d}" if not valid_dt.empty else "N/A"
    return (f"{len(df)} transactions from {d_min} to {d_max}; months: {', '.join(months)}; "
            f"categories: {', '.join(sorted(df['category'].unique()))}; currency INR; "
            f"amounts: debits are negative in the raw data, but tools report spending as positive numbers.")


def snapshot(df: pd.DataFrame) -> str:
    """Exact facts handed to the Advisor (computed by code, never by the LLM)."""
    sp = T.spending(df)
    parts = [f"Total spending (excluding savings transfers/transfers): {T.total_spent(df):,.2f}",
             f"Period: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}",
             "Monthly trend (spent, income, mom_change_pct):\n" + T.monthly_trend(df).to_string(),
             "Category breakdown (total, count, pct):\n" + T.category_breakdown(df).head(10).to_string(),
             "Category month-over-month:\n" + T.category_mom(df).head(8).to_string(),
             "Top merchants:\n" + T.top_merchants(df).to_string()]
    an = T.find_anomalies(df)
    parts.append("Unusually large debits:\n" + (an.head(5).to_string() if len(an) else "none"))
    od = T.overdraft_events(df)
    if len(od):
        parts.append(f"Negative balance: {len(od)} transactions with balance below zero; first on {od['date'].min():%Y-%m-%d}; "
                     f"lowest balance {od['balance'].min():,.2f}")
    else:
        parts.append("Negative balance: never")
    small = sp[sp["spent"] <= 500]
    parts.append(f"Small payments (500 INR or less): {len(small)} payments totalling {small['spent'].sum():,.2f}")
    sav = df[(df["type"] == "debit") & (df["category"] == "Savings Transfer")]
    parts.append(f"Savings transfers (saving, not spending): {-sav['amount'].sum():,.2f}")
    return "\n\n".join(parts)
