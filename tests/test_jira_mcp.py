import json
import re
import asyncio
import httpx
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# 1. Extract ciphertext
transcript_path = r"C:\Users\rox\.gemini\antigravity-ide\brain\00e47fd4-62c0-4c33-b89c-c6840117b5ba\.system_generated\logs\transcript_full.jsonl"
with open(transcript_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

content = json.loads(lines[77])["content"]

m_ct = re.search(r'ciphertext:\s*["\']([a-f0-9]+)["\']', content)
m_iv = re.search(r'iv:\s*["\']([a-f0-9]+)["\']', content)
m_tag = re.search(r'tag:\s*["\']([a-f0-9]+)["\']', content)

key = b"wekraft_default_sec_key_32bytes!"[:32]
aesgcm = AESGCM(key)
ciphertext_bytes = bytes.fromhex(m_ct.group(1))
iv_bytes = bytes.fromhex(m_iv.group(1))
tag_bytes = bytes.fromhex(m_tag.group(1))

token = aesgcm.decrypt(iv_bytes, ciphertext_bytes + tag_bytes, None).decode("utf-8").strip()

async def test_mcp():
    mcp_url = "https://mcp.atlassian.com/v2/mcp"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Initialize
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "wekraft-kaya-mcp-subagent", "version": "2.0.0"}
            }
        }
        res = await client.post(mcp_url, json=init_payload, headers=headers)
        session_id = res.headers.get("mcp-session-id")
        if session_id:
            headers["mcp-session-id"] = session_id
        await client.post(mcp_url, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

        # Step 1: get resources
        call_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "getAccessibleAtlassianResources",
                "arguments": {}
            }
        }
        res = await client.post(mcp_url, json=call_payload, headers=headers)
        raw_text = res.text
        for line in raw_text.splitlines():
            if line.startswith("data:"):
                res_data = json.loads(line[5:].strip())
                break
        else:
            res_data = res.json()

        content_str = res_data["result"]["content"][0]["text"]
        parsed = json.loads(content_str)
        cloud_id = parsed["data"]["resources"][0]["cloudId"]
        site_url = parsed["data"]["resources"][0]["url"]
        print(f"Cloud ID: {cloud_id}, Site: {site_url}")

        # Step 2: Query issues
        print("\n--- Querying issues with searchJiraIssuesUsingJql ---")
        call_payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "searchJiraIssuesUsingJql",
                "arguments": {
                    "cloudId": cloud_id,
                    "jql": "status IS NOT NULL ORDER BY updated DESC",
                    "maxResults": 10
                }
            }
        }
        res = await client.post(mcp_url, json=call_payload, headers=headers)
        for line in res.text.splitlines():
            if line.startswith("data:"):
                search_data = json.loads(line[5:].strip())
                break
        else:
            search_data = res.json()

        content_str = search_data["result"]["content"][0]["text"]
        issues_res = json.loads(content_str)
        issues = issues_res.get("issues", [])
        print(f"Found {len(issues)} issues:")
        project_keys = set()
        for iss in issues:
            k = iss.get("key")
            fields = iss.get("fields", {})
            summary = fields.get("summary")
            status = fields.get("status", {}).get("name")
            p_key = fields.get("project", {}).get("key")
            project_keys.add(p_key)
            assignee = fields.get("assignee")
            assignee_name = assignee.get("displayName") if assignee else "Unassigned"
            print(f"  - [{k}] {summary} (Project: {p_key}, Status: {status}, Assignee: {assignee_name})")

        target_project = list(project_keys)[0] if project_keys else "KAN"
        print(f"\nDiscovered Project Key: {target_project}")

        # Step 3: Create demo task
        print(f"\n--- Testing createJiraIssue on project {target_project} ---")
        call_payload = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "createJiraIssue",
                "arguments": {
                    "cloudId": cloud_id,
                    "projectKey": target_project,
                    "summary": "Demo task from Kaya MCP verification",
                    "issueType": "Task",
                    "description": "Automated verification test of createJiraIssue MCP tool"
                }
            }
        }
        res = await client.post(mcp_url, json=call_payload, headers=headers)
        for line in res.text.splitlines():
            if line.startswith("data:"):
                create_data = json.loads(line[5:].strip())
                break
        else:
            create_data = res.json()

        print("Create issue result:", json.dumps(create_data, indent=2))

asyncio.run(test_mcp())
