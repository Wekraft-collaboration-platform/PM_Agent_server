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

from app.agents.tools.mcp_client import execute_mcp_tool_call_session
from test_linear_sentry_task import decrypt_field_sync, LINEAR_CRED

async def main():
    linear_token = decrypt_field_sync(LINEAR_CRED)
    linear_url = "https://mcp.linear.app/mcp"

    # Create issue 'issue-2 site tracing error' in team 'Testing-new-123'
    team_id = "cd2935a6-0118-443e-826a-7bba84d6a2ed"
    
    print("Calling save_issue to create 'site tracing error'...")
    res = await execute_mcp_tool_call_session(
        linear_url,
        linear_token,
        "save_issue",
        {
            "title": "issue-2 site tracing error",
            "team": team_id,
            "description": "Site tracing error reported from project workspace.",
            "priority": 2 # High
        }
    )
    print("save_issue result:", json.dumps(res, indent=2))

    # Verify list_issues
    res_list = await execute_mcp_tool_call_session(linear_url, linear_token, "list_issues", {})
    linear_data = json.loads(res_list["content"]) if isinstance(res_list.get("content"), str) and res_list["content"].startswith("{") else res_list

    print("\nUPDATED LINEAR ISSUES:")
    for iss in linear_data.get("issues", []):
        print(f"[{iss.get('id')}] Title: '{iss.get('title')}' | Status: '{iss.get('status')}' ({iss.get('statusType')}) | Priority: {iss.get('priority')}")

if __name__ == "__main__":
    asyncio.run(main())
