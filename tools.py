"""Analysis tools: exact numbers via pandas. The LLM never does arithmetic itself.

NON_SPEND_CATS are kept out of 'spending' so the Advisor doesn't tell you that you
'overspent' on moving money into savings.
"""
from __future__ import annotations
import ast, operator as op
import pandas as pd

NON_SPEND_CATS = {"Savings Transfer", "Transfers"}


def _month(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m")


def spending(df: pd.DataFrame) -> pd.DataFrame:
    """Debits that count as real consumption."""
    d = df[(df["type"] == "debit") & (~df["category"].isin(NON_SPEND_CATS))].copy()
    d["spent"] = -d["amount"]
    return d


def total_spent(df, category: str | None = None, month: str | None = None) -> float:
    d = spending(df)
    if category: d = d[d["category"].str.lower() == category.lower()]
    if month: d = d[_month(d["date"]) == month]
    return round(float(d["spent"].sum()), 2)


def category_breakdown(df, month: str | None = None) -> pd.DataFrame:
    d = spending(df)
    if month: d = d[_month(d["date"]) == month]
    g = d.groupby("category")["spent"].agg(total="sum", count="count").sort_values("total", ascending=False)
    g["pct"] = (g["total"] / g["total"].sum() * 100).round(1)
    return g.round(2)


def monthly_trend(df) -> pd.DataFrame:
    d = spending(df)
    m = d.groupby(_month(d["date"]))["spent"].sum().round(2).to_frame("spent")
    inc = df[(df["type"] == "credit") & (df["category"] == "Income")]
    m["income"] = inc.groupby(_month(inc["date"]))["amount"].sum().round(2)
    m["income"] = m["income"].fillna(0.0)
    m["mom_change_pct"] = (m["spent"].pct_change() * 100).round(1)
    return m


def category_mom(df) -> pd.DataFrame:
    """Per-category spend by month + change between the last two months."""
    d = spending(df)
    p = d.pivot_table(index="category", columns=_month(d["date"]), values="spent", aggfunc="sum", fill_value=0).round(2)
    months = list(p.columns)
    if len(months) >= 2:
        a, b = months[-2], months[-1]
        p["change"] = (p[b] - p[a]).round(2)
        p["change_pct"] = ((p[b] - p[a]) / p[a].where(p[a] != 0) * 100).round(1)
    return p.sort_values(months[-1], ascending=False)


def top_merchants(df, n: int = 5) -> pd.DataFrame:
    d = spending(df)
    return d.groupby("merchant")["spent"].agg(total="sum", count="count").sort_values("total", ascending=False).head(n).round(2)


def find_anomalies(df, z: float = 2.0) -> pd.DataFrame:
    """Debits unusually large for their category (z-score within category, min 4 txns)."""
    d = spending(df)
    stats = d.groupby("category")["spent"].agg(["mean", "std", "count"])
    d = d.join(stats, on="category")
    d["z"] = (d["spent"] - d["mean"]) / d["std"].where(d["std"] > 0)
    out = d[(d["count"] >= 4) & (d["z"] > z)]
    out = out[["date", "description", "merchant", "category", "spent", "z"]].sort_values("z", ascending=False)
    return out.assign(spent=out["spent"].round(2), z=out["z"].round(2))


def overdraft_events(df) -> pd.DataFrame:
    return df[df["balance"] < 0][["date", "description", "amount", "balance"]]


def find_transactions(df, text: str | None = None, category: str | None = None,
                      min_amount: float | None = None, max_amount: float | None = None) -> pd.DataFrame:
    d = df.copy()
    if text: d = d[d["text"].str.contains(text, case=False, na=False)]
    if category: d = d[d["category"].str.lower() == category.lower()]
    if min_amount is not None: d = d[d["amount"].abs() >= min_amount]
    if max_amount is not None: d = d[d["amount"].abs() <= max_amount]
    return d[["date", "description", "merchant", "category", "amount"]]


# ---- Safe calculator (no eval) ----
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.USub: op.neg, ast.Mod: op.mod}

def calculator(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS: return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return round(float(ev(ast.parse(expr, mode="eval").body)), 4)
