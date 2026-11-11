"""Central config: categories + LLM factory. Switch providers with LLM_PROVIDER in .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
# explicit path next to config.py (not the current folder); utf-8-sig tolerates Notepad's BOM
ENV_LOADED = load_dotenv(ENV_PATH, override=True, encoding="utf-8-sig")


def env_report() -> str:
    """Safe diagnostics: shows where we looked and which variable NAMES exist (never values)."""
    lines = [f"looking for: {ENV_PATH}", f"file exists: {ENV_PATH.exists()}"]
    if not ENV_PATH.exists():
        near = [p.name for p in ENV_PATH.parent.iterdir() if "env" in p.name.lower()]
        lines.append(f"files named like env here: {near or 'none'}  (a '.env.txt' means Windows hid the extension)")
    else:
        from dotenv import dotenv_values
        vals = dotenv_values(ENV_PATH, encoding="utf-8-sig")
        lines.append(f"variables found: {list(vals.keys())}")
        for k, v in vals.items():
            if k.strip() != k or (v and (v != v.strip() or v[:1] in "\"'")):
                lines.append(f"  warning: '{k}' has stray spaces or quotes")
    prov = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    lines.append(f"LLM_PROVIDER = '{prov}'")
    if prov == "groq":
        key = os.getenv("GROQ_API_KEY")
    elif prov == "ollama":
        key = "n/a"
    else:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    lines.append(f"API key for '{prov}' loaded: {bool(key)}" + (f" (length {len(key)})" if key and key != "n/a" else ""))
    return "\n".join(lines)

# Spending categories the Categorizer may assign (fixed list = reliable JSON + exact analysis).
SPEND_CATEGORIES = [
    "Food & Dining", "Groceries", "Transport", "Shopping", "Utilities", "Subscriptions",
    "Entertainment", "Healthcare", "Insurance", "Education", "Fitness", "Housing",
    "Household", "Donations", "Cash Withdrawal", "Bank Charges", "Miscellaneous",
]
# Money-in / money-moving categories (kept out of "spending" by tools.py)
OTHER_CATEGORIES = ["Income", "Interest", "Cashback", "Savings Transfer", "Transfers"]
CATEGORIES = SPEND_CATEGORIES + OTHER_CATEGORIES


def get_llm(temperature: float = 0.0, json_mode: bool = True):
    """Return a LangChain chat model. LLM_PROVIDER=gemini (default) | groq | ollama.

    json_mode=True makes Groq/Ollama return valid JSON in the answer field (our chains all expect JSON).
    check_llm.py passes json_mode=False because it asks a plain-text question."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().strip("\"'").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=temperature, format="json" if json_mode else "")
    if provider == "groq":
        from langchain_groq import ChatGroq
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not found.\n" + env_report())
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        kw = {}
        # gpt-oss are reasoning models: 'low' effort keeps answers fast and stops thinking from eating the output budget.
        effort = os.getenv("GROQ_REASONING_EFFORT", "low" if "gpt-oss" in model else "").strip()
        if effort:
            kw["reasoning_effort"] = effort
        llm = ChatGroq(model=model, temperature=temperature, timeout=60, max_retries=1, **kw)
        # JSON mode: the API itself guarantees the reply is a JSON object (needs the word "JSON" in the prompt; ours has it)
        return llm.bind(response_format={"type": "json_object"}) if json_mode else llm
    from langchain_google_genai import ChatGoogleGenerativeAI
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("API key not found.\n" + env_report())
    os.environ["GOOGLE_API_KEY"] = key
    # 'gemini-flash-latest' is an alias; override with GEMINI_MODEL if your AI Studio lists another Flash model.
    # timeout + 1 SDK retry (retries on 429 also consume free-tier quota) so a bad key / model name / rate limit fails fast instead of hanging
    def make(name):
        return ChatGoogleGenerativeAI(model=name, temperature=temperature, timeout=60, max_retries=1)

    primary = make(os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
    # Optional: if the primary model is overloaded (503) or rate-limited, automatically try this one.
    fallback = os.getenv("GEMINI_FALLBACK_MODEL")
    return primary.with_fallbacks([make(fallback)]) if fallback else primary