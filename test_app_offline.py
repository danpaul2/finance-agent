"""Headless UI test with Streamlit's AppTest: scripted fake LLM + hash embeddings, no API key or network."""
import json, os, re, tempfile
from pathlib import Path
os.environ["EMBED_BACKEND"] = "hash"
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import config
import agents.categorizer as K
from agents.parser import parse_csv

K.CACHE_PATH = Path(tempfile.mkdtemp()) / "cache.json"   # never touch the real label cache
tx, _ = parse_csv("data/sample_statement.csv")
truth = {(r.description, r.merchant, r.type): r.true_category for r in tx.itertuples()}


def fake_llm(pv):
    msgs = pv.to_messages(); system, last = msgs[0].content, msgs[-1].content
    if "Categorizer Agent" in system:
        items = []
        for line in last.split("\n"):
            m = re.match(r"^(\d+) \| (\w+) \| [\d.]+ \| (.*?) \| (.*?) \|", line)
            if m:
                items.append({"id": int(m[1]), "reason": "fake", "category": truth[(m[3], m[4], m[2])]})
        return AIMessage(content=json.dumps({"items": items}))
    if "Intent Router" in system:
        q = last.lower()
        d = ("transaction_lookup", True) if "rapido" in q else ("spending_analysis", False)
        return AIMessage(content=json.dumps({"reason": "fake", "intent": d[0], "needs_retrieval": d[1]}))
    if "Transaction Lookup Agent" in system:
        ids = [l.split(" | ")[0] for l in last.split("\n") if "rapido" in l.lower()]
        return AIMessage(content=json.dumps({"relevant_ids": ids, "answer": "Here are your Rapido rides."}))
    if "Analysis Agent" in system:
        if "Observation:" not in last:
            return AIMessage(content=json.dumps({"thought": "t", "final_answer": None, "actions": [{"tool": "total_spent", "args": {}}]}))
        return AIMessage(content=json.dumps({"thought": "d", "actions": [], "final_answer": "Total spent computed."}))
    raise AssertionError(system[:50])


config.get_llm = lambda *a, **k: RunnableLambda(fake_llm)

at = AppTest.from_file("app.py", default_timeout=90).run()
assert not at.exception, at.exception
assert any("Analyze" in i.value for i in at.info), "should prompt to analyze first"

at.button(key="analyze").click().run()
assert not at.exception, at.exception
print("metrics:", {m.label: m.value for m in at.metric})
assert len(at.metric) >= 5 and len(at.tabs) == 4

at.chat_input(key="chat_box").set_value("Show me all my Rapido rides").run()
assert not at.exception, at.exception
msgs = [m for m in at.chat_message]
print("chat messages:", len(msgs))
body = " ".join(str(x.value) for m in msgs for x in m.markdown)
assert "3,435.64" in body and "Rapido" in body, body[:500]

at.chat_input(key="chat_box").set_value("How much did I spend in total?").run()
assert not at.exception, at.exception
assert "Total spent computed." in " ".join(str(x.value) for m in at.chat_message for x in m.markdown)

# dashboard period filter + prompts tab render
at.selectbox(key="period").set_value("2026-09").run()
assert not at.exception, at.exception
print("OK: UI test passed")
