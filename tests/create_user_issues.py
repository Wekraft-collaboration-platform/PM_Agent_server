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
    team_id = "cd2935a6-0118-443e-826a-7bba84d6a2ed"

    # Issue 1: firewall handling
    print("Creating 'issue-1 firewall handling'...")
    res1 = await execute_mcp_tool_call_session(
        linear_url,
        linear_token,
        "save_issue",
        {
            "title": "issue-1 firewall handling",
            "team": team_id,
            "description": "Configure and handle firewall security rules and rate limiting.",
            "priority": 2 # High
        }
    )
    print("Result 1:", json.dumps(res1, indent=2))

    # Issue 2: login auth error
    print("\nCreating 'issue-2 login auth error'...")
    res2 = await execute_mcp_tool_call_session(
        linear_url,
        linear_token,
        "save_issue",
        {
            "title": "issue-2 login auth error",
            "team": team_id,
            "description": "Authentication failure and token refresh handling on login flow.",
            "priority": 1 # Urgent
        }
    )
    print("Result 2:", json.dumps(res2, indent=2))

    # List all issues to verify
    print("\nVerifying Linear issues:")
    res_list = await execute_mcp_tool_call_session(linear_url, linear_token, "list_issues", {})
    linear_data = json.loads(res_list["content"]) if isinstance(res_list.get("content"), str) and res_list["content"].startswith("{") else res_list

    for iss in linear_data.get("issues", []):
        print(f"- [{iss.get('id')}] {iss.get('title')} | Status: {iss.get('status')} | Priority: {iss.get('priority', {}).get('name') if isinstance(iss.get('priority'), dict) else iss.get('priority')} | URL: {iss.get('url')}")

if __name__ == "__main__":
    asyncio.run(main())
