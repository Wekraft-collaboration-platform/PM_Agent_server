import uvicorn
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

load_dotenv(override=True)
print(f"[CONFIG] Environment loaded. App Name: {os.getenv('APP_NAME', 'wekraft-agent')}")

if __name__ == "__main__":
    print("[SERVER] Starting uvicorn development server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
