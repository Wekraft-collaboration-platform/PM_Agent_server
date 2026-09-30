import os
import sys
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv()

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import binascii

from app.agents.tools.mcp_client import (
    execute_mcp_tool_call_session,
    fetch_mcp_tools_list_session,
)

def decrypt_field_sync(data: dict) -> str:
    key_secret = "wekraft_default_sec_key_32bytes!"
    raw_key = key_secret.ljust(32, "0")[:32].encode("utf-8")
    aesgcm = AESGCM(raw_key)

    iv = binascii.unhexlify(data["iv"])
    ct = binascii.unhexlify(data["ciphertext"])
    tag = binascii.unhexlify(data["tag"])
    return aesgcm.decrypt(iv, ct + tag, None).decode("utf-8")

notion_cred = {
    "ciphertext": "348a716392e87aee43e86eda555be2510865135f90db20c7e65b5f8276ac5b5462b8a8f667ccd026c71d01cb1aa640edff9aaafec42806dfa26f084b7fdd9e46feb07bcb488281b315ae0472d99ace9edcdcbf7016be",
    "iv": "f152db16fd218ed4d714b078",
    "tag": "ad1a310e704c7511de5490956d3f28da",
}

async def main():
    notion_token = decrypt_field_sync(notion_cred)
    notion_url = "https://mcp.notion.com/mcp"

    tools = await fetch_mcp_tools_list_session(notion_url, notion_token)
    for t in tools:
        if t["name"] == "notion-create-pages":
            print(f"\nTool: {t['name']}")
            print(f"Description: {t.get('description')}")
            print(f"InputSchema: {json.dumps(t.get('inputSchema', {}), indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())
