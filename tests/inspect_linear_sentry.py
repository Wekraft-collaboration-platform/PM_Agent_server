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

from app.agents.tools.mcp_client import (
    execute_mcp_tool_call_session,
    fetch_mcp_tools_list_session,
)
from test_linear_sentry_task import decrypt_field_sync, LINEAR_CRED, SENTRY_CRED

async def main():
    linear_token = decrypt_field_sync(LINEAR_CRED)
    sentry_token = decrypt_field_sync(SENTRY_CRED)
    linear_url = "https://mcp.linear.app/mcp"
    sentry_url = "https://mcp.sentry.dev/mcp"

    # 1. Fetch Linear issues
    res_linear = await execute_mcp_tool_call_session(linear_url, linear_token, "list_issues", {})
    linear_data = json.loads(res_linear["content"]) if isinstance(res_linear.get("content"), str) and res_linear["content"].startswith("{") else res_linear

    print("ALL LINEAR ISSUES:")
    issues = linear_data.get("issues", [])
    for iss in issues:
        print(f"ID: {iss.get('id')} | Title: '{iss.get('title')}' | Status: {iss.get('status')} ({iss.get('statusType')}) | Priority: {iss.get('priority')} | CreatedAt: {iss.get('createdAt')}")

    # 2. Check Sentry issues
    res_sentry = await execute_mcp_tool_call_session(
        sentry_url, sentry_token, "search_issues",
        {"organizationSlug": "vrsa-solution-6q", "query": "is:unresolved", "sort": "date"}
    )
    print("\nSENTRY UNRESOLVED ISSUES:")
    print(res_sentry.get("content"))

    # 3. Check Linear tools for creating issues
    linear_tools = await fetch_mcp_tools_list_session(linear_url, linear_token)
    save_issue_tool = next((t for t in linear_tools if t["name"] in ["save_issue", "create_issue"]), None)
    print("\nSAVE_ISSUE TOOL SCHEMA:")
    print(json.dumps(save_issue_tool, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
