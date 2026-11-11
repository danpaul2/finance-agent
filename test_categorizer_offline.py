"""Offline plumbing test (no API key): a fake LLM answers from ground truth, drops one id on the
first call to exercise the repair pass. Verifies prompt formatting, LCEL chain, parsing, de-dup, merging.
It does NOT measure model accuracy - use evaluate.py with a real key for that."""
import json, re
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from agents.parser import parse_csv
from agents.categorizer import categorize, build_chain

tx, _ = parse_csv("data/sample_statement.csv")
lookup = {(r.description, r.merchant, r.type): r.true_category for r in tx.itertuples()}
calls = {"n": 0}

def fake_llm(prompt_value):
    msgs = prompt_value.to_messages()
    assert msgs[0].type == "system" and "Allowed categories" in msgs[0].content
    last = msgs[-1].content
    items = []
    for line in last.split("\n"):
        m = re.match(r"^(\d+) \| (\w+) \| [\d.]+ \| (.*?) \| (.*?) \|", line)
        if m:
            i, typ, desc, mer = int(m[1]), m[2], m[3], m[4]
            items.append({"id": i, "reason": "fake", "category": lookup[(desc, mer, typ)]})
    calls["n"] += 1
    if calls["n"] == 1 and items:
        items = items[1:]  # simulate a skipped id
    return AIMessage(content="```json\n" + json.dumps({"items": items}) + "\n```")

df = categorize(tx.drop(columns=["true_category"]), llm=RunnableLambda(fake_llm), use_rules=False, sleep=0)
acc = (df["category"] == tx["true_category"]).mean()
print("llm calls:", calls["n"], "| sources:", df["cat_source"].value_counts().to_dict(), "| plumbing accuracy:", f"{acc:.1%}")
assert df["category"].notna().all() and acc == 1.0
print("OK")
