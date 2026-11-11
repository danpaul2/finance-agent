"""Offline checks for the output-parsing safety net (empty replies, prose around JSON, <think> blocks)."""
import json
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel
from llm_utils import RobustPydanticParser, with_repair


class Out(BaseModel):
    answer: str


def make(replies, seen):
    it = iter(replies)
    def llm(pv):
        seen.append(pv.to_messages()[-1].content)
        r = next(it)
        return r if isinstance(r, AIMessage) else AIMessage(content=r)
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    return with_repair(prompt | RunnableLambda(llm) | RobustPydanticParser(pydantic_object=Out), "question")

good = json.dumps({"answer": "ok"})
for label, reply in [("clean", good), ("code fence", f"```json\n{good}\n```"), ("prose around", f"Sure! Here you go: {good} Hope that helps."),
                     ("think block", f"<think>hmm {{not json}}</think>{good}")]:
    seen = []
    assert make([reply], seen).invoke({"question": "q"}).answer == "ok", label
    assert len(seen) == 1
print("tolerant parsing: ok")

seen = []
assert make(["", "   ", good], seen).invoke({"question": "q"}).answer == "ok"
assert len(seen) == 3 and "IMPORTANT" in seen[1] and "IMPORTANT" not in seen[0]
print("empty reply -> retried with stricter instruction: ok")

seen = []
try:
    make([AIMessage(content="", response_metadata={"finish_reason": "length"})] * 3, seen).invoke({"question": "q"})
    raise SystemExit("should have failed")
except OutputParserException as e:
    assert "finish_reason=length" in str(e) and len(seen) == 3
    print("persistent empty reply -> clear error:", str(e)[:90])

seen = []
try:
    make(["not json at all"] * 3, seen).invoke({"question": "q"})
except OutputParserException as e:
    assert "not json at all" in str(e)
    print("garbage -> error shows the raw output: ok")

class RateLimit(Exception): pass
n = []
def boom(pv):
    n.append(1); raise RateLimit("429")
c = with_repair(ChatPromptTemplate.from_messages([("human", "{question}")]) | RunnableLambda(boom) | RobustPydanticParser(pydantic_object=Out), "question")
try: c.invoke({"question": "q"})
except RateLimit: pass
assert len(n) == 1
print("API errors are not retried (no wasted quota): ok")

# JSON only in the reasoning channel -> recovered; otherwise diagnostics name the extra fields
seen = []
m = AIMessage(content="", additional_kwargs={"reasoning_content": f"I think the answer is {good} done"}, response_metadata={"finish_reason": "stop"})
assert make([m], seen).invoke({"question": "q"}).answer == "ok"
m2 = AIMessage(content="", additional_kwargs={"reasoning_content": "just thinking"}, response_metadata={"finish_reason": "stop"})
try:
    make([m2] * 3, []).invoke({"question": "q"})
except OutputParserException as e:
    assert "reasoning_chars=13" in str(e)
print("reasoning-channel fallback + diagnostics: ok")

# Groq provider is wired with JSON mode (no network: only builds the object)
import os
os.environ.update(LLM_PROVIDER="groq", GROQ_API_KEY="x")
import importlib, config; importlib.reload(config)
g = config.get_llm()
assert g.kwargs.get("response_format") == {"type": "json_object"}
assert getattr(config.get_llm(json_mode=False), "reasoning_effort", None) == "low"
print("groq json_mode wiring: ok\nOK")
