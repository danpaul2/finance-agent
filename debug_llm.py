"""Shows exactly what the model sends back for a JSON request.  Run: python debug_llm.py
Paste the output here if the chat still reports an empty reply."""
from config import get_llm, env_report
print(env_report(), "\n")
llm = get_llm(json_mode=True)
msg = llm.invoke("Reply with a JSON object with keys intent and reason for this request: which category did I spend the most in")
print("content           :", repr(msg.content))
print("finish_reason     :", msg.response_metadata.get("finish_reason"))
print("extra fields      :", sorted(msg.additional_kwargs))
print("reasoning (first 200 chars):", repr(str(msg.additional_kwargs.get("reasoning_content", ""))[:200]))
print("token usage       :", msg.response_metadata.get("token_usage"))
