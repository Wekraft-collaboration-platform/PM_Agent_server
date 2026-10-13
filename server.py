import uvicorn
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

load_dotenv(override=True)


def main():
    _openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    _groq_key = os.getenv("GROQ_API_KEY", "").strip()
    print(f"[CONFIG] Environment loaded. App Name: {os.getenv('APP_NAME', 'wekraft-agent')}")
    print(f"[AUTH] Active OpenAI Key: ...{_openai_key[-4:] if len(_openai_key) >= 4 else 'Not Set'}")
    print(f"[AUTH] Active Groq Key: ...{_groq_key[-4:] if len(_groq_key) >= 4 else 'Not Set'}")
    print("[SERVER] Starting uvicorn development server on port 8080...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()

