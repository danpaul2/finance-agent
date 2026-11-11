"""Categorizer Agent: tags each transaction with a spending category.

Pipeline (cheapest first):
  1. Rule pass     - obvious merchants/keywords, zero LLM calls
  2. De-duplicate  - identical (description, merchant, method, direction) share one label
  3. LLM pass      - few-shot + mini-CoT, batched, via LCEL:  prompt | llm | PydanticOutputParser
  4. Repair pass   - any ids the model skipped or mislabelled are retried once, then fall back to Miscellaneous
"""
from __future__ import annotations
import re, time, os, json, hashlib
from pathlib import Path
from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm_utils import RobustPydanticParser, with_repair

from config import CATEGORIES
from prompts import CATEGORIZER_SYSTEM, CATEGORIZER_USER, CATEGORIZER_FEWSHOT_MESSAGES

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "category_cache.json"

CategoryName = Literal[tuple(CATEGORIES)]  # type: ignore[valid-type]


class LabelledTxn(BaseModel):
    id: int
    reason: str = Field(description="max 12 words, written before the category")
    category: CategoryName


class CategorizedBatch(BaseModel):
    items: list[LabelledTxn]


# ---------- 1. Rule pass: generic, well-known patterns (not tuned to one statement) ----------
RULES: list[tuple[str, str]] = [
    # order matters: first match wins, so specific patterns come before broad ones.
    # Word boundaries (\\b) stop matches inside longer words (e.g. 'atm' in 'treatment').
    (r"salary|payroll", "Income"),
    (r"interest (credit|paid)|int\\.pd", "Interest"),
    (r"cashback|reward", "Cashback"),
    (r"\\batm\\b|cash withdrawal", "Cash Withdrawal"),
    (r"service charge|bank charge|sms (alert )?charge|annual fee|gst on", "Bank Charges"),
    (r"subscription|netflix|spotify|prime video|hotstar|youtube|google one", "Subscriptions"),
    (r"insurance|\\blic\\b", "Insurance"),
    (r"swiggy|zomato|restaurant|\\bcafe\\b|coffee|bakery|dominos|mcdonald", "Food & Dining"),
    (r"uber|\\bola\\b|rapido|metro|irctc|fuel|petrol|indian oil|auto rickshaw|bus pass|bmtc", "Transport"),
    (r"bigbasket|blinkit|zepto|supermarket|grocery", "Groceries"),
    (r"pharmacy|apollo|diagnostic|hospital|clinic|dental|doctor|practo", "Healthcare"),
    (r"bwssb|electricity|bescom|water bill|mobile recharge|broadband|airtel|\\bjio\\b", "Utilities"),
    (r"\\brent\\b", "Housing"),
    (r"bookmyshow|\\bpvr\\b|inox|cinema|movie", "Entertainment"),
    (r"amazon|flipkart|myntra|ajio", "Shopping"),
]


def rule_category(row) -> str | None:
    text = f"{row['description']} {row['merchant']}".lower()
    if "savings" in text and row["type"] == "debit":
        return "Savings Transfer"
    for pattern, cat in RULES:
        if re.search(pattern, text):
            # direction guard: credits can only be Income/Interest/Cashback/Transfers
            if row["type"] == "credit" and cat not in ("Income", "Interest", "Cashback"):
                return None
            return cat
    return None


# ---------- 3. LLM chain (LCEL) ----------
def build_chain(llm):
    parser = RobustPydanticParser(pydantic_object=CategorizedBatch)
    prompt = ChatPromptTemplate.from_messages(
        [("system", CATEGORIZER_SYSTEM), *CATEGORIZER_FEWSHOT_MESSAGES, ("human", CATEGORIZER_USER)]
    ).partial(format_instructions=parser.get_format_instructions())
    # re-asks (with a stricter instruction) only when the reply is unparsable; API errors are not retried
    return with_repair(prompt | llm | parser, "transactions")


def _prompt_hash() -> str:
    """Cache is invalidated automatically whenever the prompts change."""
    blob = CATEGORIZER_SYSTEM + CATEGORIZER_USER + repr(CATEGORIZER_FEWSHOT_MESSAGES)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=0))


def _fmt(i: int, r) -> str:
    return f"{i} | {r['type']} | {abs(r['amount']):.2f} | {r['description']} | {r['merchant']} | {r['upi_id'] or ''} | {r['method'] or ''}"


def _run_batches(chain, items: list[tuple[int, pd.Series]], batch_size: int, sleep: float, on_batch=None) -> dict[int, str]:
    out: dict[int, str] = {}
    n_batches = -(-len(items) // batch_size)
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        text = "\n".join(_fmt(i, r) for i, r in chunk)
        t0 = time.time()
        print(f"  batch {s // batch_size + 1}/{n_batches}: sending {len(chunk)} items...", flush=True)
        try:
            res = chain.invoke({"transactions": text})
        except Exception as e:
            raise RuntimeError(f"LLM call failed on batch {s // batch_size + 1}: {type(e).__name__}: {str(e)[:300]}") from e
        print(f"  batch {s // batch_size + 1}/{n_batches}: done in {time.time() - t0:.1f}s", flush=True)
        valid = {i for i, _ in chunk}
        part = {it.id: it.category for it in res.items if it.id in valid}
        out.update(part)
        if on_batch:
            on_batch(part)
        time.sleep(sleep)
    return out


def categorize(df: pd.DataFrame, llm=None, use_rules: bool = True, batch_size: int = 30,
               sleep: float | None = None, progress=None) -> pd.DataFrame:
    """Return df with `category` and `cat_source` ('rule' | 'llm' | 'cache' | 'fallback').

    Labels from the default configured LLM are cached in data/category_cache.json, so re-running
    never re-spends API quota on transactions already labelled. The cache is skipped when a custom
    `llm` is passed in (e.g. the offline test's fake model), so test data can never pollute it.
    Delete the file to force fresh labels."""
    df = df.copy()
    sleep = float(os.getenv("CATEGORIZER_SLEEP", "1.0")) if sleep is None else sleep
    df["category"], df["cat_source"] = None, None

    if use_rules:
        for idx, row in df.iterrows():
            c = rule_category(row)
            if c:
                df.at[idx, "category"], df.at[idx, "cat_source"] = c, "rule"

    todo = df[df["category"].isna()]
    if len(todo):
        use_cache = llm is None
        cache = _load_cache() if use_cache else {}
        ph = _prompt_hash()
        ckey = lambda k: ph + "|" + "|".join(map(str, k))
        # 2. de-duplicate identical transaction "shapes" so each is labelled once
        key = lambda r: (r["description"], r["merchant"], r["method"], r["type"])
        groups: dict[tuple, list] = {}
        for idx, row in todo.iterrows():
            groups.setdefault(key(row), []).append(idx)
        reps = [(n, df.loc[idxs[0]]) for n, idxs in enumerate(groups.values())]
        keys = list(groups.keys())

        labels = {n: cache[ckey(k)] for n, k in enumerate(keys) if ckey(k) in cache}
        cached_ids = set(labels)
        pending = [(n, r) for n, r in reps if n not in labels]
        if cached_ids:
            print(f"  cache: {len(cached_ids)}/{len(keys)} transaction shapes already labelled", flush=True)

        def remember(part: dict[int, str]):
            if use_cache:
                for n, cat in part.items():
                    cache[ckey(keys[n])] = cat
                _save_cache(cache)  # saved after every batch, so a later failure loses nothing

        if pending:
            if llm is None:
                from config import get_llm
                llm = get_llm()
            chain = build_chain(llm)
            labels.update(_run_batches(chain, pending, batch_size, sleep, on_batch=remember))
            missing = [(n, r) for n, r in pending if n not in labels]
            if missing:  # 4. repair pass, once
                labels.update(_run_batches(chain, missing, batch_size, sleep, on_batch=remember))

        for n, k in enumerate(keys):
            cat = labels.get(n, "Miscellaneous")
            src = ("cache" if n in cached_ids else "llm") if n in labels else "fallback"
            for idx in groups[k]:
                df.at[idx, "category"], df.at[idx, "cat_source"] = cat, src
            if progress: progress((n + 1) / len(keys))
    return df
