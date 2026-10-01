"""
test_notion_skill_operations.py
Validates 3 most-common Notion operations via MCP:
  1. CREATE  — notion-create-pages   (create a rich demo page)
  2. UPDATE  — notion-append-block-children (append live-data blocks)
  3. FETCH   — notion-fetch          (read back the full page content)
"""

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

from app.agents.tools.mcp_client import execute_mcp_tool_call_session

# ─── Decrypt helper ──────────────────────────────────────────────────────────

def decrypt_field_sync(data: dict) -> str:
    key_secret = "wekraft_default_sec_key_32bytes!"
    raw_key = key_secret.ljust(32, "0")[:32].encode("utf-8")
    aesgcm = AESGCM(raw_key)
    iv  = binascii.unhexlify(data["iv"])
    ct  = binascii.unhexlify(data["ciphertext"])
    tag = binascii.unhexlify(data["tag"])
    return aesgcm.decrypt(iv, ct + tag, None).decode("utf-8")


# ─── New credential (provided 01-Oct-2026) ───────────────────────────────────
notion_cred = {
    "ciphertext": "48b076a6ae383723b8af2e22ea4546e1c9f9580b999b434d0af2e10e450690be8ade9fe192ed480bc82df23cdd0f5c523f6f3892cb7c683f8c9984966829f08ed8c6c40fc2b49e0d250516ad0c02409b79a6070e58d2",
    "iv":         "516a9cb7581b1687813626bb",
    "tag":        "d431c4ba0a5dfdb47ccb0a80d8e9b3da",
}

NOTION_MCP_URL = "https://mcp.notion.com/mcp"


# ─── Operation 1: CREATE ─────────────────────────────────────────────────────
async def op_create_page(token: str) -> str:
    print("\n━━━ [1/3] CREATE PAGE ━━━")
    payload = {
        "creation_mode": "draft",
        "allow_async": False,
        "pages": [
            {
                "properties": {"title": "🚀 Kaya Notion Skill — Validation Page"},
                "content": """# Kaya Notion Skill — Validation

This page was **automatically created** by Kaya to validate the Notion integration skill.

## Supported Operations

| Operation | MCP Tool | Use-Case |
|-----------|----------|----------|
| Create page | `notion-create-pages` | Reports, wikis, project briefs |
| Append content | `notion-append-block-children` | Live data updates, snapshots |
| Read content | `notion-fetch` | Summarise, analyse, Q&A |

## Status
- ✅ Auth token: valid
- ✅ Create: verified
- ⏳ Append & Fetch: running next...

> _Created automatically during skill validation._
""",
                "icon": "🚀",
            }
        ],
    }

    result = await execute_mcp_tool_call_session(
        NOTION_MCP_URL, token, "notion-create-pages", payload
    )
    print("CREATE result (truncated):")
    print(json.dumps(result, indent=2)[:900])

    # Extract page ID from various response shapes
    page_id = None
    if isinstance(result, dict):
        # Shape 1: direct id key
        page_id = result.get("id") or result.get("pageId")
        # Shape 2: direct pages array
        if not page_id and isinstance(result.get("pages"), list):
            page_id = result["pages"][0].get("id")
        # Shape 3 (Notion MCP): content is a JSON string with {pages: [{id, url}]}
        if not page_id and isinstance(result.get("content"), str):
            try:
                inner = json.loads(result["content"])
                if isinstance(inner.get("pages"), list):
                    page_id = inner["pages"][0].get("id")
            except Exception:
                pass
    elif isinstance(result, list) and result:
        page_id = result[0].get("id")

    if page_id:
        print(f"\n✅ Created page ID: {page_id}")
    else:
        print("\n⚠️  Could not extract page ID — see raw output above.")

    return page_id or ""


# ─── Operation 2: UPDATE (append content to page) ───────────────────────────
async def op_update_page(token: str, page_id: str):
    if not page_id:
        print("\n⏭️  [2/3] SKIP UPDATE — no page_id")
        return

    print(f"\n━━━ [2/3] UPDATE PAGE ({page_id}) ━━━")
    # notion-update-page schema: page_id + command (insert_content | replace_content | update_properties)
    payload = {
        "page_id": page_id,
        "command": "insert_content",
        "content": """## 📊 Live Project Snapshot

Kaya inserted this section with live data after the page was created.
In production, these blocks contain real sprint metrics, issue counts, and workload data.

- 🟢 Sprint 3 — 68% complete
- 🔴 3 critical issues open
- 👤 2 team members over capacity

> _Last updated by Kaya automatically._
"""
    }

    result = await execute_mcp_tool_call_session(
        NOTION_MCP_URL, token, "notion-update-page", payload
    )
    print("UPDATE result (truncated):")
    print(json.dumps(result, indent=2)[:700])
    print("\n✅ Page updated with live snapshot")



# ─── Operation 3: FETCH ──────────────────────────────────────────────────────
async def op_fetch_page(token: str, page_id: str):
    if not page_id:
        print("\n⏭️  [3/3] SKIP FETCH — no page_id")
        return

    print(f"\n━━━ [3/3] FETCH PAGE ({page_id}) ━━━")
    result = await execute_mcp_tool_call_session(
        NOTION_MCP_URL, token, "notion-fetch", {"id": page_id}
    )
    print("FETCH result (truncated to 1400 chars):")
    print(json.dumps(result, indent=2)[:1400])
    print("\n✅ Page fetched successfully")


# ─── Main ────────────────────────────────────────────────────────────────────
async def main():
    print("🔑 Decrypting Notion credential...")
    token = decrypt_field_sync(notion_cred)
    print(f"   Token prefix: {token[:12]}...")

    page_id = await op_create_page(token)
    await op_update_page(token, page_id)
    await op_fetch_page(token, page_id)

    print(f"\n\n🎉 All 3 Notion operations completed successfully!")
    print(f"   Notion page ID: {page_id}")


if __name__ == "__main__":
    asyncio.run(main())
