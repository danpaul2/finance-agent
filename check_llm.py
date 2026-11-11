"""One tiny LLM call. Run this first: python check_llm.py
Tells you immediately whether the key, model name and network work."""
import time
from config import get_llm, env_report
print(env_report(), "\n")
llm = get_llm(json_mode=False)
print("model:", getattr(llm, "model", "?"))
t = time.time()
try:
    r = llm.invoke("Reply with exactly: OK")
    print("response:", r.content, f"({time.time() - t:.1f}s)")
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:500])
