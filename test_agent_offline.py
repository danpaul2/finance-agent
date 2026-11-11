"""Offline end-to-end test of routing + RAG + ReAct + advisor/reflection.
Uses a scripted fake LLM and a hash embedding, so it needs no API key, no network, no model download.
It checks the plumbing and the exactness of numbers, NOT the quality of a real model's answers."""
import json, re
import chromadb
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

import tools as T
import rag
from agents.parser import parse_csv
from router import FinanceAgent

tx, _ = parse_csv("data/sample_statement.csv")
df = tx.rename(columns={"true_category": "category"})  # stand-in for the Categorizer output
calls = []


def fake_llm(pv):
    msgs = pv.to_messages()
    system, last = msgs[0].content, msgs[-1].content
    calls.append(system[:30])
    if "Intent Router" in system:
        q = last.lower()
        if "cab" in q or "ride" in q:
            d = ("transaction_lookup", True)
        elif "how much" in q or "balance" in q:
            d = ("spending_analysis", False)
        else:
            d = ("savings_guidance", True)
        return AIMessage(content=json.dumps({"reason": "fake", "intent": d[0], "needs_retrieval": d[1]}))
    if "Analysis Agent" in system:
        if "Observation:" not in last:
            return AIMessage(content=json.dumps({"thought": "need shopping total", "final_answer": None,
                             "actions": [{"tool": "total_spent", "args": {"category": "Shopping"}}]}))
        val = re.search(r"Observation: ([\d.]+)", last).group(1)
        return AIMessage(content=json.dumps({"thought": "done", "actions": [], "final_answer": f"You spent {val} on Shopping."}))
    if "Transaction Lookup Agent" in system:
        ids = [l.split(" | ")[0] for l in last.split("\n") if re.search(r"cab|rapido|ola|uber", l, re.I)]
        return AIMessage(content=json.dumps({"relevant_ids": ids, "answer": "Here are your cab rides."}))
    if "Advisor Agent" in system:
        return AIMessage(content=json.dumps({"reasoning": "fake", "answer": "Shopping is high at 99,999 INR. Cut it."}))
    if "fact-checker" in system:
        return AIMessage(content=json.dumps({"unsupported_claims": ["99,999"], "revised_answer": "Shopping is your top category. Cut it."}))
    raise AssertionError("unexpected prompt: " + system[:60])


ctx = rag.RagContext.create(df, client=chromadb.EphemeralClient(), ef=rag.HashEmbedding())
print("indexed:", ctx.txns.count(), "transactions,", ctx.guide.count(), "guidance chunks")
assert ctx.txns.count() == len(df) and ctx.guide.count() >= 15

# vector search sanity (keyword-hash embedding, so overlap-based)
g = ctx.search_guidance("reduce subscriptions and streaming costs", n=2)
print("guidance top hit:", g[0]["section"])
assert any("ubscription" in x["text"] for x in g)
f = ctx.search_transactions("shopping", n=5, category="Shopping")
assert f and all(h["category"] == "Shopping" for h in f)  # metadata filter works

agent = FinanceAgent(df, RunnableLambda(fake_llm), ctx)

r = agent.ask("Show me all my cab rides")
print("\n[lookup]", r.intent, r.route_source, "|", r.answer.splitlines()[-1]); print("\n".join("  ", ) if False else "", *r.trace, sep="\n  - ")
assert r.intent == "transaction_lookup" and r.tables["transactions"]
exp = sum(h["amount"] for h in r.tables["transactions"])
assert f"{abs(exp):,.2f}" in r.answer

r = agent.ask("How much did I spend on shopping?")
print("\n[analysis]", r.intent, "|", r.answer); print(*r.trace, sep="\n  - ")
assert r.intent == "spending_analysis" and f"{T.total_spent(df, 'Shopping')}" in r.answer

n = len(calls)
r = agent.ask("How can I save more money?")
print("\n[guidance]", r.intent, r.route_source, "|", r.answer); print(*r.trace, sep="\n  - ")
assert r.intent == "savings_guidance" and r.route_source == "rule" and "99,999" not in r.answer and r.sources
assert len(calls) - n == 2  # advisor + reflection (router skipped by rule shortcut)

n = len(calls)
r = agent.ask("thanks!")
assert r.intent == "out_of_scope" and len(calls) == n  # zero LLM calls
print("\nOK: all routes passed")
