"""Answers computed from the statement when the language model does not return a usable one.

Every figure comes from the pandas tools (and the calculator for differences and percentages).
The wording follows the question: a food total, a month-to-month comparison, a balance check,
and so on, instead of a generic failure line.
"""
from __future__ import annotations
import calendar
import re
import pandas as pd

import tools as T
from agents.analysis import _CATEGORY_ALIASES, _available_months, canon_category, canon_month

_REFUSAL = re.compile(
    r"\b(could not|couldn't|couldnt|cannot|can't|cant|unable to|not able to|"
    r"was not able|wasn't able|i (?:don't|do not) (?:know|have)|"
    r"no (?:information|data) available|try rephras)\b",
    re.I,
)
_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_STOP = {
    "show", "list", "find", "give", "tell", "what", "which", "when", "where", "how",
    "much", "many", "did", "does", "was", "were", "have", "has", "the", "and", "for",
    "all", "any", "from", "with", "that", "this", "your", "you", "are", "about",
    "spend", "spent", "spending", "total", "totals", "transaction", "transactions",
    "payment", "payments", "purchase", "purchases", "ride", "rides", "fare", "fares",
    "statement", "account", "bank", "upi", "inr", "rupee", "rupees", "please", "month",
    "category", "categories", "money", "cost", "costs", "between", "during", "into",
    "over", "under", "more", "less", "than", "most", "least", "ever", "negative",
    "balance", "went", "compare", "comparison", "versus", "vs",
}


def looks_like_refusal(text: str | None) -> bool:
    """True when the model declined instead of stating a figure from the statement."""
    if not text or not text.strip():
        return True
    if text.strip().lower() in {"null", "none", "n/a", "undefined"}:
        return True
    if not _REFUSAL.search(text):
        return False
    return not bool(re.search(r"₹|\b\d", text))


def answer_usable(answer: str | None, scratch: str) -> bool:
    """Keep a model answer only when it is not a refusal and its figures appear in tool output."""
    if looks_like_refusal(answer):
        return False
    claimed = _big_numbers(answer or "")
    if not claimed:
        return True
    evidence = _big_numbers(scratch or "")
    return all(any(abs(n - s) <= max(1.0, 0.01 * abs(s)) for s in evidence) for n in claimed)


def _big_numbers(text: str) -> list[float]:
    out = []
    for raw in _NUM.findall(text or ""):
        try:
            n = float(raw.replace(",", ""))
        except ValueError:
            continue
        if n < 100 or (1900 <= n <= 2100 and n == int(n)):
            continue
        out.append(n)
    return out


def inr(x: float) -> str:
    return f"\u20b9{abs(float(x)):,.2f}"


def pretty_month(ym: str) -> str:
    dt = pd.to_datetime(str(ym), errors="coerce")

    if pd.isna(dt):
        return str(ym)

    return dt.strftime("%B %Y")


def mentioned_months(df: pd.DataFrame, question: str) -> list[str]:
    """Months named in the question, in the order they appear, resolved against the statement."""
    available = _available_months(df)
    if not available:
        return []
    ql = question.lower()
    found: list[tuple[int, str]] = []
    for m in re.finditer(r"20\d{2}-\d{2}", question):
        ym = canon_month(df, m.group(0))
        if ym in available:
            found.append((m.start(), ym))
    for token, num in sorted(_month_tokens().items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(rf"\b{token}\b", ql):
            hits = [ym for ym in available if int(ym[5:7]) == num]
            years = re.findall(r"20\d{2}", question)
            if years:
                hits = [h for h in hits if h[:4] in years] or hits
            if hits:
                found.append((m.start(), hits[-1]))
    if re.search(r"\b(this month|last month|latest month|previous month|prior month)\b", ql):
        which = available[-1]
        if re.search(r"\b(previous month|prior month)\b", ql) and len(available) >= 2:
            which = available[-2]
        found.append((len(question), which))
    found.sort(key=lambda item: item[0])
    out = []
    for _, ym in found:
        if ym not in out:
            out.append(ym)
    return out


def _month_tokens() -> dict[str, int]:
    tokens = {}
    for i in range(1, 13):
        tokens[calendar.month_name[i].lower()] = i
        tokens[calendar.month_abbr[i].lower()] = i
    tokens["sept"] = 9
    return tokens


def mentioned_categories(df: pd.DataFrame, question: str) -> list[str]:
    ql = re.sub(r"[^a-z0-9]+", " ", question.lower())
    labels = [str(c) for c in pd.unique(df["category"].dropna())]
    found: list[tuple[int, str]] = []
    padded = f" {ql} "
    for label in labels:
        norm = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
        pos = padded.find(f" {norm} ")
        if pos != -1:
            found.append((pos, label))
    for alias, target in _CATEGORY_ALIASES.items():
        if f" {alias} " not in padded:
            continue
        canon = canon_category(df, target)
        if canon and all(canon != label for _, label in found):
            found.append((padded.find(f" {alias} "), canon))
    found.sort(key=lambda item: item[0])
    out = []
    for _, label in found:
        if label not in out:
            out.append(label)
    return out


def mentioned_merchant(df: pd.DataFrame, question: str) -> str | None:
    ql = question.lower()
    names = sorted({str(m).strip() for m in df["merchant"].dropna().unique() if str(m).strip()}, key=len, reverse=True)
    for name in names:
        if len(name) < 3:
            continue
        if re.search(rf"\b{re.escape(name.lower())}\b", ql):
            return name
    return None


def matching_transactions(df: pd.DataFrame, question: str) -> list[dict]:
    """Every statement row that literally matches a merchant, category, or keyword in the question."""
    d = df
    merchant = mentioned_merchant(df, question)
    categories = mentioned_categories(df, question)
    if merchant:
        d = d[d["merchant"].str.lower() == merchant.lower()]
    elif categories:
        d = d[d["category"].isin(categories)]
    else:
        tokens = [t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) >= 3 and t not in _STOP]
        hit = None
        for tok in sorted(tokens, key=len, reverse=True):
            rows = df[df["text"].str.contains(re.escape(tok), case=False, na=False)]
            if len(rows):
                hit = rows
                break
        if hit is None:
            return []
        d = hit
    months = mentioned_months(df, question)
    if months:
        ym = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m")
        d = d[ym.isin(months)]
    rows = []
    ordered = d.sort_values("date") if len(d) else d
    for r in ordered.itertuples():
        when = pd.to_datetime(r.date, errors="coerce")
        rows.append({
            "id": str(r.txn_id),
            "date": when.strftime("%Y-%m-%d") if pd.notna(when) else str(r.date),
            "merchant": r.merchant,
            "description": r.description,
            "category": r.category,
            "amount": float(r.amount),
        })
    return rows


def _period(months: list[str], df: pd.DataFrame) -> str:
    if len(months) == 1:
        return pretty_month(months[0])
    if len(months) >= 2:
        return f"{pretty_month(months[0])} to {pretty_month(months[-1])}"
    available = _available_months(df)
    if not available:
        return "the statement"
    if len(available) == 1:
        return pretty_month(available[0])
    return f"{pretty_month(available[0])} to {pretty_month(available[-1])}"


def _spend_frame(df: pd.DataFrame, category: str | None, month: str | None) -> pd.DataFrame:
    d = T.spending(df)
    if category:
        d = d[d["category"].str.lower() == category.lower()]
    if month:
        d = d[pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m") == month]
    return d


def _asks(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text, re.I))


def grounded_analysis(df: pd.DataFrame, question: str) -> str:
    """Plain-language answer for a spending question, with every figure computed by code."""
    ql = question.lower()
    months = mentioned_months(df, question)
    categories = mentioned_categories(df, question)
    merchant = mentioned_merchant(df, question)

    if _asks(r"\b(negative|overdraft|below zero|went negative|in the red)\b", ql) or (
            _asks(r"\bbalance\b", ql) and _asks(r"\b(zero|negative|lowest|minimum|min)\b", ql)):
        return _overdraft(df)
    if _asks(r"\b(grew|grow|growth|increas|decreas|declin|fell|dropped|shr[au]nk|month[- ]over[- ]month|changed the most)\b", ql):
        return _growth(df, months, categories, ql)
    if _asks(r"\b(anomal|unusual|unusually|outlier|largest debit|biggest (purchase|debit|transaction)|largest transaction)\b", ql):
        return _anomalies(df)
    if _asks(r"\b(top merchant|merchants|who did i pay|where did i (spend|shop))\b", ql):
        return _merchants(df)
    if merchant and categories and merchant.lower() in " ".join(categories).lower():
        merchant = None
    if merchant and _asks(r"\b(how much|total|spent|spend|cost|paid)\b", ql):
        return _merchant_total(df, merchant, months)
    if len(categories) >= 2 and _asks(r"\b(more|less|compar|versus|vs|than|difference|higher|lower)\b", ql):
        return _compare_categories(df, categories[:2], months)
    if _asks(r"\b(income|salary|earned|earn|credited to me)\b", ql) and not _asks(r"\bspend", ql):
        return _income(df, months)
    if _asks(r"\b(saved|savings transfer|moved to savings|put into savings)\b", ql):
        return _savings(df)
    if not categories and _asks(r"\b(breakdown|by category|where (did|does)( my| the)? money|spend the most|spent the most|biggest categor|largest categor)\b", ql):
        return _breakdown(df, months[:1])
    if not categories and _asks(r"\b(each month|every month|month by month|monthly trend|over time|trend)\b", ql):
        return _trend(df)
    if categories or merchant or _asks(r"\b(how much|total|spent|spend|average|avg)\b", ql):
        return _spend(df, categories[0] if categories else None, months, ql)
    return _overview(df, question)


def _overdraft(df: pd.DataFrame) -> str:
    od = T.overdraft_events(df)
    lowest = float(df["balance"].min())
    if od.empty:
        return (f"No. The balance never went below zero. "
                f"The lowest balance in the statement is {inr(lowest)}.")
    first = pd.to_datetime(od["date"]).min()
    return (f"Yes. The balance went below zero on {len(od)} transactions, "
            f"first on {first:%d %b %Y}. The lowest balance was {inr(float(od['balance'].min()))}.")


def _growth(df: pd.DataFrame, months: list[str], categories: list[str], ql: str) -> str:
    available = _available_months(df)
    if len(available) < 2:
        return "The statement only covers one month, so there is no earlier month to compare."
    if len(months) >= 2:
        left, right = months[0], months[1]
    elif len(months) == 1:
        idx = available.index(months[0]) if months[0] in available else len(available) - 1
        if idx == 0:
            return f"{pretty_month(months[0])} is the first month in the statement, so there is no earlier month to compare."
        left, right = available[idx - 1], months[0]
    else:
        left, right = available[-2], available[-1]
    want_drop = bool(re.search(r"\b(fell|dropped|decreas|declin|shr[au]nk|down the most)\b", ql))
    table = _category_totals(df, [left, right])
    if categories:
        return _one_category_growth(table, categories[0], left, right)
    if table.empty:
        return f"There is no category spending to compare between {pretty_month(left)} and {pretty_month(right)}."
    ranked = []
    for category, row in table.iterrows():
        a, b = float(row[left]), float(row[right])
        delta = float(T.calculator(f"{b}-{a}"))
        pct = None if a == 0 else float(T.calculator(f"({b}-{a})/{a}*100"))
        ranked.append((category, a, b, delta, pct))
    ranked.sort(key=lambda item: item[3], reverse=not want_drop)
    cat, a, b, delta, pct = ranked[0]
    direction = "fell" if delta < 0 else "grew"
    if want_drop:
        direction = "fell the most" if delta < 0 else "changed the least upward"
    else:
        direction = "grew the most" if delta > 0 else "fell the least"
    pct_bit = ""
    if pct is not None:
        pct_bit = f" ({pct:+.1f}%)"
    elif a == 0 and b:
        pct_bit = f" (no spending in {pretty_month(left)}, so a percentage change is not meaningful)"
    return (f"Between {pretty_month(left)} and {pretty_month(right)}, **{cat}** {direction}: "
            f"{inr(a)} to {inr(b)}, a change of {'+' if delta >= 0 else '-'}{inr(delta)}{pct_bit}. "
            f"Savings transfers and transfers between accounts are excluded.")


def _one_category_growth(table: pd.DataFrame, category: str, left: str, right: str) -> str:
    if category not in table.index:
        return (f"{category} has no spending in {pretty_month(left)} or {pretty_month(right)}. "
                f"Spending totals exclude savings transfers and transfers between accounts.")
    a, b = float(table.loc[category, left]), float(table.loc[category, right])
    delta = float(T.calculator(f"{b}-{a}"))
    if a == 0:
        pct_bit = "There was no spending in the first of those months, so a percentage is not meaningful."
    else:
        pct = float(T.calculator(f"({b}-{a})/{a}*100"))
        pct_bit = f"That is {pct:+.1f}%."
    word = "up" if delta >= 0 else "down"
    return (f"{category} went from {inr(a)} in {pretty_month(left)} to {inr(b)} in {pretty_month(right)}, "
            f"{word} {'+' if delta >= 0 else '-'}{inr(delta)}. {pct_bit}")


def _category_totals(df: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    d = T.spending(df)
    d["month"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m")
    d = d[d["month"].isin(months)]
    if d.empty:
        return pd.DataFrame(columns=months)
    p = d.pivot_table(index="category", columns="month", values="spent", aggfunc="sum", fill_value=0.0)
    for m in months:
        if m not in p.columns:
            p[m] = 0.0
    return p[months].round(2)


def _anomalies(df: pd.DataFrame) -> str:
    an = T.find_anomalies(df)
    if an.empty:
        return "No debit stands out as unusually large for its category."
    lines = [f"There are {len(an)} unusually large debits for their category. The largest ones:"]
    for r in an.head(5).itertuples():
        when = pd.to_datetime(r.date)
        lines.append(f"- {when:%d %b %Y}: {r.merchant} ({r.category}) {inr(r.spent)}")
    return "\n".join(lines)


def _merchants(df: pd.DataFrame) -> str:
    top = T.top_merchants(df, 5)
    if top.empty:
        return "There is no merchant spending in this statement."
    lines = ["The merchants you spent the most with (savings transfers excluded):"]
    for merchant, row in top.iterrows():
        lines.append(f"- {merchant}: {inr(row['total'])} across {int(row['count'])} payments")
    return "\n".join(lines)


def _merchant_total(df: pd.DataFrame, merchant: str, months: list[str]) -> str:
    d = df[(df["type"] == "debit") & (df["merchant"].str.lower() == merchant.lower())].copy()
    if months:
        ym = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m")
        d = d[ym.isin(months if len(months) > 1 else months)]
    paid = round(float((-d["amount"]).sum()), 2) if len(d) else 0.0
    where = _period(months, df) if months else "the whole statement"
    if not len(d):
        return f"There are no payments to {merchant} in {where}."
    return (f"You paid {merchant} {inr(paid)} across {len(d)} transactions in {where}.")


def _compare_categories(df: pd.DataFrame, categories: list[str], months: list[str]) -> str:
    month = months[0] if len(months) == 1 else None
    a_name, b_name = categories
    a = float(T.total_spent(df, a_name, month))
    b = float(T.total_spent(df, b_name, month))
    diff = float(T.calculator(f"{a}-{b}"))
    where = pretty_month(month) if month else "the whole statement"
    if diff == 0:
        relation = "the same as"
    elif diff > 0:
        relation = f"{inr(diff)} more than"
    else:
        relation = f"{inr(diff)} less than"
    return (f"In {where}, {a_name} is {inr(a)} and {b_name} is {inr(b)}. "
            f"{a_name} is {relation} {b_name}.")


def _income(df: pd.DataFrame, months: list[str]) -> str:
    d = df[(df["type"] == "credit") & (df["category"] == "Income")]
    if months:
        ym = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m")
        d = d[ym.isin(months)]
    total = round(float(d["amount"].sum()), 2) if len(d) else 0.0
    where = _period(months, df) if months else "the whole statement"
    return f"Income in {where} is {inr(total)} across {len(d)} credits."


def _savings(df: pd.DataFrame) -> str:
    sav = df[(df["type"] == "debit") & (df["category"] == "Savings Transfer")]
    total = round(float((-sav["amount"]).sum()), 2) if len(sav) else 0.0
    return (f"You moved {inr(total)} into savings across {len(sav)} transfers. "
            f"Those transfers are saving, not spending, so they are left out of spending totals.")


def _breakdown(df: pd.DataFrame, months: list[str]) -> str:
    month = months[0] if months else None
    g = T.category_breakdown(df, month)
    where = pretty_month(month) if month else "the whole statement"
    if g.empty:
        return f"There is no spending in {where}."
    top = g.head(5)
    lines = [f"Spending by category in {where} (savings transfers excluded). "
             f"The largest is **{top.index[0]}** at {inr(top.iloc[0]['total'])} ({top.iloc[0]['pct']}% of spending):"]
    for category, row in top.iterrows():
        lines.append(f"- {category}: {inr(row['total'])} ({row['pct']}%, {int(row['count'])} transactions)")
    return "\n".join(lines)


def _trend(df: pd.DataFrame) -> str:
    m = T.monthly_trend(df)
    if m.empty:
        return "There is no monthly spending to show."
    lines = ["Spending and income by month (savings transfers excluded from spending):"]
    prev = None
    for month, row in m.iterrows():
        change = ""
        if prev is not None and prev != 0:
            pct = float(T.calculator(f"({row['spent']}-{prev})/{prev}*100"))
            change = f", {pct:+.1f}% versus the previous month"
        lines.append(f"- {pretty_month(str(month))}: spent {inr(row['spent'])}, income {inr(row['income'])}{change}")
        prev = float(row["spent"])
    return "\n".join(lines)


def _spend(df: pd.DataFrame, category: str | None, months: list[str], ql: str) -> str:
    month = months[0] if len(months) == 1 else None
    if category in {"Income", "Interest", "Cashback"}:
        return _credits(df, category, months)
    if category in {"Savings Transfer", "Transfers"} and not _asks(r"\bspend", ql):
        return _savings(df) if category == "Savings Transfer" else _transfer_total(df, months)
    if len(months) >= 2 and category:
        bits = [f"{pretty_month(m)}: {inr(T.total_spent(df, category, m))}" for m in months]
        return f"{category} spending — " + "; ".join(bits) + "."
    if len(months) >= 2 and category is None:
        bits = [f"{pretty_month(m)}: {inr(T.total_spent(df, None, m))}" for m in months]
        return "Spending by month — " + "; ".join(bits) + ". Savings transfers are excluded."
    total = float(T.total_spent(df, category, month))
    rows = _spend_frame(df, category, month)
    where = _period(months if month else [], df)
    subject = category or "overall"
    if _asks(r"\b(average|avg|per month)\b", ql):
        available = months or _available_months(df)
        n = max(len(available), 1)
        avg = float(T.calculator(f"{total}/{n}"))
        scope = f"for {category}" if category else "overall"
        return (f"Average monthly spending {scope} is {inr(avg)} "
                f"({inr(total)} across {n} month{'s' if n != 1 else ''}, {where}).")
    if total == 0:
        return (f"There is no {subject} spending in {where}. "
                f"Spending totals exclude savings transfers and transfers between accounts.")
    return (f"In {where}, {subject} spending is {inr(total)} across {len(rows)} transactions. "
            f"Savings transfers and transfers between accounts are not included.")


def _credits(df: pd.DataFrame, category: str, months: list[str]) -> str:
    d = df[(df["type"] == "credit") & (df["category"] == category)]
    if months:
        ym = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m")
        d = d[ym.isin(months)]
    total = round(float(d["amount"].sum()), 2) if len(d) else 0.0
    where = _period(months, df) if months else "the whole statement"
    return f"{category} in {where} is {inr(total)} across {len(d)} credits."


def _transfer_total(df: pd.DataFrame, months: list[str]) -> str:
    d = df[(df["type"] == "debit") & (df["category"] == "Transfers")]
    if months:
        ym = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m")
        d = d[ym.isin(months)]
    total = round(float((-d["amount"]).sum()), 2) if len(d) else 0.0
    where = _period(months, df) if months else "the whole statement"
    return (f"Transfers to other people in {where} total {inr(total)} across {len(d)} payments. "
            f"They are excluded from spending totals.")


def _overview(df: pd.DataFrame, question: str) -> str:
    total = float(T.total_spent(df))
    g = T.category_breakdown(df)
    top = ""
    if not g.empty:
        top = f" The largest category is {g.index[0]} at {inr(g.iloc[0]['total'])} ({g.iloc[0]['pct']}% of spending)."
    od = T.overdraft_events(df)
    balance = (f" The balance went negative on {len(od)} transactions, down to {inr(float(od['balance'].min()))}."
               if len(od) else " The balance never went negative.")
    return (f"Across {_period([], df)}, total spending is {inr(total)}."
            f"{top}{balance} Savings transfers are excluded from spending.")


def grounded_guidance(df: pd.DataFrame, question: str, guidance: list[dict] | None = None) -> str:
    """Savings notes from computed facts, used when the advisor model does not answer."""
    g = T.category_breakdown(df)
    total = float(T.total_spent(df))
    lines = [f"Across {_period([], df)} you spent {inr(total)}, excluding savings transfers."]
    if not g.empty:
        lines.append("The categories taking the largest share:")
        for category, row in g.head(3).iterrows():
            lines.append(f"- {category}: {inr(row['total'])} ({row['pct']}%)")
    mom = T.category_mom(df)
    if "change" in getattr(mom, "columns", []):
        rising = mom[mom["change"] > 0].sort_values("change", ascending=False).head(1)
        if len(rising):
            cat = rising.index[0]
            lines.append(f"{cat} rose the most into the latest month, by {inr(float(rising.iloc[0]['change']))}.")
    od = T.overdraft_events(df)
    if len(od):
        lines.append(f"The balance went negative on {len(od)} transactions (lowest {inr(float(od['balance'].min()))}). "
                     "Rebuilding a small buffer before discretionary spending would stop that.")
    sp = T.spending(df)
    small = sp[sp["spent"] <= 500]
    if len(small):
        lines.append(f"Payments of {inr(500)} or less: {len(small)} of them, totalling {inr(float(small['spent'].sum()))}. "
                     "A weekly cap on those is the most direct cut.")
    if guidance:
        lines.append("From the budgeting notes: " + guidance[0].get("text", "").strip()[:400])
    lines.append("Practical steps: pause the top discretionary category for the rest of this month, "
                 "cancel a subscription you have not used, and move the usual savings transfer the day income arrives.")
    return "\n".join(lines)
