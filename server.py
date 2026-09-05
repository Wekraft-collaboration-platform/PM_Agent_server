import uvicorn
import os
from dotenv import load_dotenv

load_dotenv(override=True)
print(f"[CONFIG] OpenAI Key Loaded: ...{os.getenv('OPENAI_API_KEY')[-4:]}")


if __name__ == "__main__":
    print("[SERVER] Starting uvicorn...")
    # Run from app.main:app
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
