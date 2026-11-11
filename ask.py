"""Terminal chat for the whole pipeline (before the Streamlit UI exists):  python ask.py"""
from agents.parser import parse_csv
from agents.categorizer import categorize
from config import get_llm
import rag
from router import FinanceAgent

tx, _ = parse_csv("data/sample_statement.csv")
print("Categorizing (cached labels are free)...")
df = categorize(tx.drop(columns=["true_category"]))
print("Building ChromaDB indexes (first run downloads a small embedding model)...")
ctx = rag.RagContext.create(df)
agent = FinanceAgent(df, get_llm(), ctx)
print("Ready. Type a question, or 'quit'.\n")
while True:
    q = input("you> ").strip()
    if q.lower() in ("quit", "exit", ""):
        break
    try:
        r = agent.ask(q)
    except Exception as e:  # noqa: BLE001
        print(f"error: {type(e).__name__}: {str(e)[:800]}\n")
        continue
    print("\n".join("  · " + t for t in r.trace))
    print(f"\n{r.answer}\n")
