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
    "sentry": [
        "sentry", "error", "exception", "crash", "stacktrace", "issue", "bug", "alert", "incident"
    ],
    "hubspot": [
        "hubspot", "crm", "contact", "deal", "lead", "company", "marketing", "pipeline", "sales"
    ],
    "vercel": [
        "vercel", "deployment", "deploy", "domain", "project", "env", "environment", "build", "alias", "preview"
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPLIANT MCP STREAMABLE HTTP & SSE PROTOCOL CLIENT
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPLIANT MCP STREAMABLE HTTP & SSE PROTOCOL CLIENT
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_mcp_tools_list_session(
    mcp_url: str,
    access_token: str,
    timeout_sec: float = 8.0,
) -> List[Dict[str, Any]]:
    """
    Performs MCP initialize handshake and queries 'tools/list' to discover all live tools exposed by the MCP server.
    """
    token = access_token.strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=True) as client:
        try:
            # Step 1: Initialize MCP session
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "wekraft-kaya-mcp-discover", "version": "1.0.0"},
                },
            }
            init_resp = await client.post(mcp_url, headers=headers, json=init_payload)
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

            # Step 3: Query tools/list
            list_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
            list_resp = await client.post(mcp_url, headers=headers, json=list_payload)
            list_resp.raise_for_status()

            content_type = list_resp.headers.get("content-type", "")
            data: Dict[str, Any] = {}

            if "text/event-stream" in content_type:
                for line in list_resp.text.splitlines():
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            break
                        except Exception:
                            pass
            else:
                data = list_resp.json()

            tools = data.get("result", {}).get("tools", [])
            return tools if isinstance(tools, list) else []

        except Exception as e:
            print(f"[MCP CLIENT] ⚠️ Could not fetch tools/list from {mcp_url}: {e}")
            return []


async def execute_mcp_tool_call_session(
    mcp_url: str,
    access_token: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 12.0,
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
            session_id = init_resp.headers.get("mcp-session-id")
            if session_id:
                headers["mcp-session-id"] = session_id

            try:
                await client.post(
                    mcp_url,
                    headers=headers,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
            except Exception:
                pass

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
# 2. CONVEX PROJECT MCP CONNECTIONS & SMART PRUNING
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
    """Strictly filters available connections to ONLY the connectors mentioned or requested in the user query."""
    if not connections:
        return []

    q_lower = user_query.lower()

    # Priority 1: Direct connector name match in query (e.g. "linear", "jira", "slack", "notion", "calendly")
    explicit_matches = [
        c for c in connections
        if c.get("connectorId", "").lower() in q_lower
    ]
    if explicit_matches:
        # User explicitly asked for specific app(s) -> ONLY return those
        return explicit_matches

    # Priority 2: Keyword match
    matched_connectors = []
    for conn in connections:
        c_id = conn.get("connectorId", "").lower()
        keywords = CONNECTOR_KEYWORDS.get(c_id, [c_id])
        if any(kw in q_lower for kw in keywords):
            matched_connectors.append(conn)

    return matched_connectors if matched_connectors else connections


# ─────────────────────────────────────────────────────────────────────────────
# 3. MASTER ReAct WORKFLOW EXECUTION (Max 3 Retries, Dynamic Tool Calling)
# ─────────────────────────────────────────────────────────────────────────────

async def execute_mcp_agent_workflow(
    project_id: str,
    user_query: str,
    user_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes dynamic ReAct MCP Sub-Agent workflow for Kaya:
    1. Filters connectors to strictly those asked by user (e.g. Linear / Jira only, no Slack).
    2. Dynamically queries live tools/list (e.g., 66 Linear tools, Notion tools, etc.).
    3. Uses LLM ReAct loop (Max 3 iterations) to choose the right tool, inspect output, retry if needed.
    4. Produces zero-hallucination structured Markdown findings for Kaya Synthesizer.
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

    default_mcp_urls = {
        "linear": "https://mcp.linear.app/mcp",
        "jira": "https://mcp.atlassian.com/v2/mcp",
        "notion": "https://mcp.notion.com/mcp",
        "slack": "https://mcp.slack.com/mcp",
        "calendly": "https://mcp.calendly.com",
        "sentry": "https://mcp.sentry.dev/mcp",
        "hubspot": "https://mcp.hubspot.com",
        "vercel": "https://mcp.vercel.com",
    }

    # Discover live tools across strictly scoped connectors
    tool_registry: Dict[str, Dict[str, Any]] = {}
    openai_tools: List[Dict[str, Any]] = []

    for conn in target_connections:
        c_id = conn.get("connectorId", "").lower()
        token = conn.get("accessToken", "")
        meta = conn.get("metadata", {}) or {}
        mcp_url = meta.get("mcpUrl") or default_mcp_urls.get(c_id, f"https://mcp.{c_id}.com/mcp")
        workspace_name = meta.get("workspaceName", f"{c_id.capitalize()} Workspace")

        if c_id == "jira":
            # Register Jira search tool
            tool_name = "jira_search_issues"
            tool_registry[tool_name] = {
                "connector_id": "jira",
                "mcp_url": mcp_url,
                "token": token,
                "workspace_name": workspace_name,
                "is_jira": True,
                "orig_name": "searchJiraIssuesUsingJql",
            }
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Searches Jira tickets, issues, epics, and tasks in Atlassian workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query or context"}
                        },
                    },
                },
            })
            continue

        # Fetch live tools list from MCP server (e.g. 66 Linear tools!)
        discovered_tools = await fetch_mcp_tools_list_session(mcp_url, token)
        if not discovered_tools and meta.get("tools"):
            discovered_tools = [{"name": t, "description": f"Tool {t} on {c_id}", "inputSchema": {}} for t in meta.get("tools", [])]

        for t in discovered_tools:
            raw_name = t.get("name", "")
            if not raw_name:
                continue
            # Keep tool identifier safe for OpenAI function calling
            fn_name = f"{c_id}__{raw_name}".replace("-", "_").replace(".", "_")
            desc = t.get("description", f"Call {raw_name} on {c_id.capitalize()}")
            schema = t.get("inputSchema", {}) or {"type": "object", "properties": {}}

            tool_registry[fn_name] = {
                "connector_id": c_id,
                "mcp_url": mcp_url,
                "token": token,
                "workspace_name": workspace_name,
                "is_jira": False,
                "orig_name": raw_name,
            }
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": fn_name,
                    "description": desc[:500],
                    "parameters": schema,
                },
            })

    if not openai_tools:
        return {
            "summary": f"⚠️ Connected to {', '.join(connected_app_names)}, but no accessible tools were discovered.",
            "connected_apps": connected_app_names,
            "executed_tools": [],
        }

    # Initialize ReAct LLM with gpt-4.1-mini (handles large 66+ tool schemas without token limits)
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    mcp_model = os.getenv("MCP_AGENT_MODEL", "gpt-4.1-mini")

    llm = ChatOpenAI(
        model=mcp_model,
        openai_api_key=openai_api_key,
        temperature=0.0,
        max_retries=2,
    )

    llm_with_tools = llm.bind_tools(openai_tools)

    react_system_prompt = (
        f"You are the specialized MCP Integration ReAct Agent for WEKRAFT.\n"
        f"Active connected apps strictly scoped for this query: {', '.join(connected_app_names)}.\n"
        f"Goal: Retrieve the live third-party data required to answer the user's query.\n"
        f"Rules:\n"
        f"1. Select and call the most relevant tool from the available tools (e.g. for notifications call notification tools, for issues call issue tools).\n"
        f"2. IMPORTANT for tool arguments: Do NOT pass internal display names into third-party 'assignee' or 'user' filters (e.g. Linear/Jira/Slack expect UUIDs or emails, not local app nicknames). Fetch issues/items with general filters (e.g. limit: 20 or orderBy: updatedAt) so valid workspace data is not filtered out.\n"
        f"3. Inspect tool output: if a filtered call returns 0 results or empty data, immediately retry without strict filters (e.g. list_issues with just limit: 20).\n"
        f"4. Maximum 3 tool calls total. Never hallucinate data — report exact findings from the tool responses."
    )

    messages: List[Any] = [
        SystemMessage(content=react_system_prompt),
        HumanMessage(content=f"User Query: '{user_query}'"),
    ]

    executed_tools: List[str] = []
    tool_results_log: List[str] = []
    MAX_RETRIES = 3

    for step in range(MAX_RETRIES):
        try:
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # Check if LLM requested tool calls
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                # LLM is satisfied and provided final response
                break

            # Execute tool call(s)
            for tc in tool_calls:
                fn_name = tc.get("name")
                args = tc.get("args") or {}
                call_id = tc.get("id", f"call_{step}")

                tool_meta = tool_registry.get(fn_name)
                if not tool_meta:
                    tool_output = f"Error: Tool '{fn_name}' not found in registry."
                else:
                    orig_name = tool_meta["orig_name"]
                    c_id = tool_meta["connector_id"]
                    w_name = tool_meta["workspace_name"]
                    executed_tools.append(orig_name)
                    print(f"[MCP AGENT] 🛠️ Calling '{orig_name}' on {c_id.capitalize()} ({w_name}) with args: {args}...")

                    # Emit live SSE event to frontend UI
                    try:
                        from langgraph.config import get_stream_writer
                        writer = get_stream_writer()
                        writer({"tool_called": orig_name, "caller": "MCP Agent"})
                    except Exception:
                        pass

                    if tool_meta["is_jira"]:
                        res = await execute_mcp_jira_workflow(
                            mcp_url=tool_meta["mcp_url"],
                            access_token=tool_meta["token"],
                            user_query=user_query,
                        )
                        tool_output = res.get("content") or res.get("error") or "No data returned."
                    else:
                        res = await execute_mcp_tool_call_session(
                            mcp_url=tool_meta["mcp_url"],
                            access_token=tool_meta["token"],
                            tool_name=orig_name,
                            arguments=args,
                        )
                        if "error" in res:
                            tool_output = f"Error: {res['error']}"
                        else:
                            tool_output = res.get("content", "Empty result.")

                    print(f"[MCP AGENT] ✅ {c_id.capitalize()} returned {len(str(tool_output))} chars.")
                    tool_results_log.append(f"- **{c_id.capitalize()} ({orig_name})**:\n```text\n{str(tool_output)[:1500]}\n```")

                messages.append(ToolMessage(content=str(tool_output)[:4000], tool_call_id=call_id))

        except Exception as err:
            print(f"[MCP AGENT] ⚠️ ReAct Step {step + 1} Error: {err}")
            break

    # Build final summary for Kaya
    lines = [f"### Live MCP Data from {', '.join(connected_app_names)}:"]
    if tool_results_log:
        lines.extend(tool_results_log)
    else:
        lines.append(f"ℹ️ Queried {', '.join(connected_app_names)} but no matching data was found for this query.")

    return {
        "summary": "\n".join(lines),
        "connected_apps": connected_app_names,
        "executed_tools": executed_tools,
    }
