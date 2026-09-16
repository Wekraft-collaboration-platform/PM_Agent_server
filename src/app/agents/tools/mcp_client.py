"""
mcp_client.py — Native Model Context Protocol (MCP) Client & Tool Orchestrator for Kaya.

Handles:
1. Dynamic Token Retrieval from Convex (strictly scoped per projectId for Kaya's connectors).
2. Compliant MCP Protocol Handshake (initialize -> notifications/initialized -> tools/call).
3. Session ID header persistence (mcp-session-id).
4. Smart Tool Pruning based on user query intent & active connectors.
5. Atlassian Rovo MCP 2-step execution (getAccessibleAtlassianResources -> searchJiraIssuesUsingJql).
6. Minimal, clean console logging (only important MCP tool details).
7. Strict Zero-Hallucination output formatting for Kaya Synthesizer.
"""

import os
import json
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from app.agents.tools.tools import convex_post_async

load_dotenv()

# Connector category keywords for intent-based Smart Tool Pruning
CONNECTOR_KEYWORDS: Dict[str, List[str]] = {
    "slack": [
        "slack", "message", "channel", "chat", "team", "dm", "post", "send message", "thread", "conversation", "huddle"
    ],
    "calendly": [
        "calendly", "schedule", "meeting", "booking", "availability", "slot", "invite", "calendar link", "event type", "reschedule"
    ],
    "linear": [
        "linear", "ticket", "issue", "cycle", "roadmap", "backlog item", "linear sprint", "linear task"
    ],
    "notion": [
        "notion", "doc", "page", "prd", "spec", "wiki", "workspace doc", "database", "notes", "rfc"
    ],
    "jira": [
        "jira", "epic", "story", "board", "jira ticket", "jira issue", "atlassian"
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPLIANT MCP STREAMABLE HTTP & SSE PROTOCOL CLIENT
# ─────────────────────────────────────────────────────────────────────────────

async def execute_mcp_tool_call_session(
    mcp_url: str,
    access_token: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 10.0,
) -> Dict[str, Any]:
    """
    Performs full MCP Streamable HTTP session:
    1. 'initialize' handshake with protocolVersion '2024-11-05'.
    2. 'notifications/initialized' notification.
    3. 'tools/call' invocation with session ID persistence.
    """
    token = access_token.strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
        # Step 1: Initialize MCP session
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "wekraft-kaya-agent", "version": "1.0.0"},
            },
        }

        try:
            init_resp = await client.post(mcp_url, headers=headers, json=init_payload)
            # Capture mcp-session-id if provided
            session_id = init_resp.headers.get("mcp-session-id")
            if session_id:
                headers["mcp-session-id"] = session_id

            # Step 2: Send notifications/initialized
            try:
                await client.post(
                    mcp_url,
                    headers=headers,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
            except Exception:
                pass

            # Step 3: Invoke tool via tools/call
            call_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {},
                },
            }

            call_resp = await client.post(mcp_url, headers=headers, json=call_payload)
            call_resp.raise_for_status()

            content_type = call_resp.headers.get("content-type", "")
            data: Dict[str, Any] = {}

            if "text/event-stream" in content_type:
                for line in call_resp.text.splitlines():
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            break
                        except Exception:
                            pass
                if not data:
                    data = {"raw_text": call_resp.text}
            else:
                data = call_resp.json()

            if "result" in data:
                content_items = data["result"].get("content", [])
                extracted_text = []
                for item in content_items:
                    if isinstance(item, dict) and "text" in item:
                        extracted_text.append(item["text"])
                    elif isinstance(item, str):
                        extracted_text.append(item)

                full_content = "\n".join(extracted_text) if extracted_text else str(data["result"])
                return {"content": full_content, "raw": data["result"], "isError": False}

            if "error" in data:
                return {"error": data["error"].get("message", str(data["error"])), "isError": True}

            return {"content": str(data), "isError": False}

        except Exception as err:
            return {"error": str(err), "isError": True}


async def execute_mcp_jira_workflow(
    mcp_url: str,
    access_token: str,
    user_query: str = "",
    timeout_sec: float = 12.0,
) -> Dict[str, Any]:
    """
    Dedicated Atlassian MCP Orchestrator:
    1. Performs 'initialize' handshake and retrieves `mcp-session-id`.
    2. Calls `getAccessibleAtlassianResources` to get accessible cloudId.
    3. Calls `searchJiraIssuesUsingJql` to retrieve live Jira issues/tasks.
    """
    token = access_token.strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
        try:
            # 1. Initialize session
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "wekraft-kaya-jira", "version": "1.0.0"},
                },
            }
            init_resp = await client.post(mcp_url, headers=headers, json=init_payload)
            init_resp.raise_for_status()

            session_id = init_resp.headers.get("mcp-session-id")
            if session_id:
                headers["mcp-session-id"] = session_id

            # 2. Get Accessible Resources
            res_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "getAccessibleAtlassianResources",
                    "arguments": {},
                },
            }
            res_resp = await client.post(mcp_url, headers=headers, json=res_payload)
            res_resp.raise_for_status()

            cloud_id = None
            res_text = res_resp.text
            for line in res_text.splitlines():
                if line.startswith("data:"):
                    try:
                        p = json.loads(line[5:].strip())
                        content = p.get("result", {}).get("content", [])
                        if content and "text" in content[0]:
                            parsed_inner = json.loads(content[0]["text"])
                            resources = parsed_inner.get("data", {}).get("resources", [])
                            if resources and "cloudId" in resources[0]:
                                cloud_id = resources[0]["cloudId"]
                                break
                    except Exception:
                        pass

            if not cloud_id:
                cloud_id = "e56da97a-1da4-40fd-bed2-4c2663ef28e7"

            # 3. Call searchJiraIssuesUsingJql
            jql = "created >= -365d order by created DESC"
            search_payload = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "searchJiraIssuesUsingJql",
                    "arguments": {
                        "cloudId": cloud_id,
                        "jql": jql,
                    },
                },
            }

            search_resp = await client.post(mcp_url, headers=headers, json=search_payload)
            search_resp.raise_for_status()

            # Parse issues
            issues_list = []
            for line in search_resp.text.splitlines():
                if line.startswith("data:"):
                    try:
                        p = json.loads(line[5:].strip())
                        content = p.get("result", {}).get("content", [])
                        if content and "text" in content[0]:
                            raw_data = json.loads(content[0]["text"])
                            issues_list = raw_data.get("data", {}).get("issues", [])
                            break
                    except Exception:
                        pass

            if issues_list:
                formatted_issues = []
                for iss in issues_list:
                    key = iss.get("key", "")
                    fields = iss.get("fields", {})
                    summary = fields.get("summary", "")
                    status = fields.get("status", {}).get("name", "Unknown")
                    issue_type = fields.get("issuetype", {}).get("name", "Task")
                    priority = fields.get("priority", {}).get("name", "Normal")
                    assignee_obj = fields.get("assignee")
                    assignee = assignee_obj.get("displayName", "Unassigned") if isinstance(assignee_obj, dict) else "Unassigned"
                    formatted_issues.append(
                        f"• [{key}] {summary} (Type: {issue_type} | Status: {status} | Priority: {priority} | Assignee: {assignee})"
                    )

                text_summary = f"Total Jira Issues Found: {len(issues_list)}\n" + "\n".join(formatted_issues)
                return {"content": text_summary, "isError": False, "count": len(issues_list)}
            else:
                return {"content": "Zero Jira issues found for this workspace.", "isError": False, "count": 0}

        except Exception as e:
            return {"error": f"Jira MCP call error: {str(e)}", "isError": True}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVEX PROJECT MCP CONNECTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_project_mcp_connections_async(project_id: str) -> List[Dict[str, Any]]:
    """Fetches active MCP connections with decrypted tokens from Convex."""
    if not project_id:
        return []
    try:
        data = await convex_post_async("getProjectMcpConnections", {"projectId": project_id})
        connections = data.get("connections", [])
        return [
            c for c in connections
            if c.get("agent") == "kaya" or c.get("connectorId", "").lower() in CONNECTOR_KEYWORDS
        ]
    except Exception as e:
        print(f"[MCP CLIENT ERROR] Failed to fetch connections for project {project_id}: {e}")
        return []


def smart_prune_connectors(user_query: str, connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filters available connections based on user query intent keywords."""
    if not connections:
        return []

    q_lower = user_query.lower()
    matched_connectors = []

    for conn in connections:
        c_id = conn.get("connectorId", "").lower()
        keywords = CONNECTOR_KEYWORDS.get(c_id, [c_id])
        if any(kw in q_lower for kw in keywords):
            matched_connectors.append(conn)

    return matched_connectors if matched_connectors else connections


# ─────────────────────────────────────────────────────────────────────────────
# 3. MASTER WORKFLOW EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

async def execute_mcp_agent_workflow(
    project_id: str,
    user_query: str,
    user_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes MCP Sub-Agent workflow for Kaya:
    1. Loads project connections & decrypted tokens from Convex.
    2. Calls remote MCP tools with full session initialization.
    3. Emits clean, minimal console output.
    """
    if not project_id:
        return {
            "summary": "⚠️ No active project specified for MCP integrations.",
            "connected_apps": [],
            "executed_tools": [],
        }

    connections = await fetch_project_mcp_connections_async(project_id)
    active_connections = [c for c in connections if c.get("accessToken")]

    if not active_connections:
        return {
            "summary": (
                "ℹ️ **No Connected MCP Integrations**: No third-party tools (Linear, Notion, Slack, Calendly, Jira) "
                "are currently connected to this workspace. Connect tools in the **Integrations** tab."
            ),
            "connected_apps": [],
            "executed_tools": [],
        }

    target_connections = smart_prune_connectors(user_query, active_connections)
    connected_app_names = [c.get("connectorId", "unknown").capitalize() for c in target_connections]

    lines = [f"### MCP Data from Connected Workspaces ({', '.join(connected_app_names)}):"]
    executed_tools = []

    for conn in target_connections:
        c_id = conn.get("connectorId", "").lower()
        token = conn.get("accessToken", "")
        meta = conn.get("metadata", {}) or {}
        
        default_mcp_urls = {
            "linear": "https://mcp.linear.app/mcp",
            "jira": "https://mcp.atlassian.com/v2/mcp",
            "notion": "https://mcp.notion.com/mcp",
            "slack": "https://mcp.slack.com/mcp",
            "calendly": "https://mcp.calendly.com",
        }
        mcp_url = meta.get("mcpUrl") or default_mcp_urls.get(c_id, f"https://mcp.{c_id}.com/mcp")
        workspace_name = meta.get("workspaceName", f"{c_id.capitalize()} Workspace")
        tools_list = meta.get("tools", [])

        if c_id == "jira":
            target_tool = "searchJiraIssuesUsingJql"
            executed_tools.append(target_tool)
            print(f"[MCP AGENT] 🛠️ Calling '{target_tool}' on Jira ({workspace_name})...")

            res = await execute_mcp_jira_workflow(
                mcp_url=mcp_url,
                access_token=token,
                user_query=user_query,
            )

            lines.append(f"- **Jira Integration** ({workspace_name}):")
            if res.get("isError"):
                print(f"[MCP AGENT] ⚠️ Jira MCP Error: {res.get('error')}")
                lines.append(f"  • ⚠️ Could not fetch from Jira MCP: {res.get('error')}.")
                lines.append(f"  • *Zero items retrieved. Do not invent any issues.*")
            else:
                content_str = res.get("content", "").strip()
                count = res.get("count", 0)
                print(f"[MCP AGENT] ✅ Jira returned {count} issues.")
                lines.append(f"  • **Live Jira Tasks & Issues** ({count} retrieved):\n```text\n{content_str}\n```")
            continue

        # Map target tool for other connectors
        target_tool = None
        args: Dict[str, Any] = {}

        if c_id == "linear":
            target_tool = "list_issues" if "list_issues" in tools_list else "linear_list_issues"
            if not tools_list:
                target_tool = "list_issues"
            args = {"limit": 20}
        elif c_id == "notion":
            target_tool = "search" if "search" in tools_list else "notion_search"
            if not tools_list:
                target_tool = "search"
            args = {"query": user_query}
        elif c_id == "slack":
            target_tool = "list_channels" if "list_channels" in tools_list else "slack_list_channels"
        elif c_id == "calendly":
            target_tool = "get_availability" if "get_availability" in tools_list else "calendly_get_availability"

        if target_tool:
            executed_tools.append(target_tool)
            print(f"[MCP AGENT] 🛠️ Calling '{target_tool}' on {c_id.capitalize()} ({workspace_name})...")

            res = await execute_mcp_tool_call_session(
                mcp_url=mcp_url,
                access_token=token,
                tool_name=target_tool,
                arguments=args,
            )

            lines.append(f"- **{c_id.capitalize()} Integration** ({workspace_name}):")

            if "error" in res:
                err_msg = res["error"]
                if "401" in err_msg or "invalid_token" in err_msg:
                    err_msg = "OAuth token expired or unauthorized. Please disconnect and reconnect in Integrations tab."
                print(f"[MCP AGENT] ⚠️ {c_id.capitalize()} MCP Response: {err_msg}")
                lines.append(f"  • ⚠️ {err_msg}")
                lines.append(f"  • *Zero items retrieved. Do not invent any issues.*")
            elif "content" in res and res["content"]:
                content_str = res["content"].strip()
                print(f"[MCP AGENT] ✅ {c_id.capitalize()} returned live data ({len(content_str)} chars).")
                lines.append(f"  • **Live Data Received**:\n```text\n{content_str}\n```")
            else:
                print(f"[MCP AGENT] ℹ️ {c_id.capitalize()} returned empty result.")
                lines.append(f"  • *Zero active items found in {workspace_name}.*")

    summary_text = "\n".join(lines)
    return {
        "summary": summary_text,
        "connected_apps": connected_app_names,
        "executed_tools": executed_tools,
    }
