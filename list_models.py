"""Lists the models YOUR key can actually use, for the provider set in .env. Run: python list_models.py
Copy a name from the output into .env (GEMINI_MODEL=... or GROQ_MODEL=...)."""
import json, os, urllib.request, urllib.error
import config  # loads .env

provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def get(url, headers=None):
    req = urllib.request.Request(url, headers={"User-Agent": "finance-agent/1.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:300]}")
    except Exception as e:
        raise SystemExit(f"{type(e).__name__}: {e}")


SKIP = ("image", "tts", "live", "audio", "embedding", "robotics", "whisper", "guard", "transcribe", "orpheus")

if provider == "groq":
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit(config.env_report())
    data = get("https://api.groq.com/openai/v1/models", {"Authorization": f"Bearer {key}"})
    rows = sorted(m["id"] for m in data.get("data", []) if not any(x in m["id"].lower() for x in SKIP))
    print("Text models available to your Groq key (set GROQ_MODEL=<name>):")
    for n in rows:
        print("  ", n)
elif provider == "gemini":
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit(config.env_report())
    names, token = [], None
    while True:
        data = get(f"https://generativelanguage.googleapis.com/v1beta/models?pageSize=100&key={key}" + (f"&pageToken={token}" if token else ""))
        names += [m["name"].removeprefix("models/") for m in data.get("models", [])
                  if "generateContent" in m.get("supportedGenerationMethods", [])]
        token = data.get("nextPageToken")
        if not token:
            break
    print("Text models available to your Gemini key (set GEMINI_MODEL=<name>):")
    for n in sorted(x for x in names if x.startswith("gemini") and not any(s in x for s in SKIP)):
        print("  ", n, "   <- flash family" if "flash" in n else "")
else:
    print(f"Provider '{provider}' has no listing here. For Ollama run: ollama list")
