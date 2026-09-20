"""
mcp_client.py — Generic Multi-App Model Context Protocol (MCP) Sub-Agent Orchestrator for Kaya.

Architecture:
- Zero Hardcoded Tools or Per-App Step Sequences.
- Parallel Fan-Out / Fan-In Sub-Workers via asyncio.gather().
- Generic Tool Classification: splits app tools into 'discovery' (0 required args) vs 'action' (has required args).
- Schema & Context Isolation: each app runs its own sub-agent loop without tool pollution.
- Structured Finalization (finalize_result) enforcing accurate data extraction and zero hallucination.
- Live SSE Stream Emission for frontend progress tracking.
- Output formatting for Kaya Executive PM Synthesizer.
"""

import os
import json
import time
import httpx
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.agents.tools.tools import convex_post_async

load_dotenv()
logger = logging.getLogger("mcp_orchestrator")

DEFAULT_STEP_BUDGET = 5
DEFAULT_WORKER_TIMEOUT_S = 45.0
FINALIZE_TOOL = "finalize_result"

# Default fallback MCP endpoints for known providers
DEFAULT_MCP_URLS: Dict[str, str] = {
    "linear": "https://mcp.linear.app/mcp",
    "jira": "https://mcp.atlassian.com/v2/mcp",
    "notion": "https://mcp.notion.com/mcp",
    "slack": "https://mcp.slack.com/mcp",
    "calendly": "https://mcp.calendly.com",
    "sentry": "https://mcp.sentry.dev/mcp",
    "hubspot": "https://mcp.hubspot.com",
    "vercel": "https://mcp.vercel.com",
    "github": "https://mcp.github.com",
}

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
    "github": [
        "github", "pull request", "pr", "repo", "repository", "commit", "branch", "release"
    ],
}


def get_default_mcp_url(connector_id: str) -> str:
    """Returns standard default MCP URL for given connector ID or standard fallback pattern."""
    cid = connector_id.lower().strip()
    return DEFAULT_MCP_URLS.get(cid, f"https://mcp.{cid}.com/mcp")


class WorkerResult(BaseModel):
    connector: str
    ok: bool
    error: Optional[str] = None
    data: List[Dict[str, Any]] = Field(default_factory=list)
    tools_executed: List[str] = Field(default_factory=list)
    resolved_context: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. MCP PROTOCOL CLIENT (Streamable HTTP & SSE)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_mcp_tools_list_session(
    mcp_url: str,
    access_token: str,
    timeout_sec: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Performs MCP initialize handshake and queries 'tools/list' to discover live tools.
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
                    "clientInfo": {"name": "wekraft-kaya-mcp-orchestrator", "version": "2.0.0"},
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
            logger.warning(f"[MCP CLIENT] Could not fetch tools/list from {mcp_url}: {e}")
            return []


async def execute_mcp_tool_call_session(
    mcp_url: str,
    access_token: str,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 15.0,
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
                "clientInfo": {"name": "wekraft-kaya-mcp-subagent", "version": "2.0.0"},
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


async def _call_tool_with_retry(
    mcp_url: str,
    token: str,
    tool_name: str,
    args: Dict[str, Any],
    retries: int = 2,
    base_delay: float = 0.5,
) -> Dict[str, Any]:
    """Retries a tool call with exponential backoff on network failures."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            res = await execute_mcp_tool_call_session(
                mcp_url=mcp_url,
                access_token=token,
                tool_name=tool_name,
                arguments=args,
            )
            if not res.get("isError"):
                return res
            last_err = res.get("error", "Unknown tool error")
        except Exception as e:
            last_err = e
        if attempt < retries:
            await asyncio.sleep(base_delay * (2 ** attempt))

    return {"error": f"Tool '{tool_name}' failed after {retries + 1} attempts: {last_err}", "isError": True}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONVEX INTEGRATIONS & GENERIC ROUTING
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_project_mcp_connections_async(project_id: str) -> List[Dict[str, Any]]:
    """Fetches active MCP connections with decrypted tokens from Convex for the active project."""
    if not project_id:
        return []
    try:
        data = await convex_post_async("getProjectMcpConnections", {"projectId": project_id})
        connections = data.get("connections", [])
        return [
            c for c in connections
            if c.get("accessToken") and (c.get("agent") == "kaya" or c.get("connectorId", "").lower() in CONNECTOR_KEYWORDS)
        ]
    except Exception as e:
        logger.error(f"[MCP CLIENT ERROR] Failed to fetch connections for project {project_id}: {e}")
        return []


def smart_prune_connectors(user_query: str, active_connections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generic routing matching query tokens against connectorId, displayName, and tokens.
    Derives tokens at runtime from connection objects without relying on noisy stopwords.
    Fails open (returns all active connections) if no specific app is named, so broad queries succeed.
    """
    if not active_connections:
        return []

    q = user_query.lower()
    matched = []
    generic_stopwords = {"workspace", "monitoring", "deployments", "integration", "app", "tools", "issue", "task", "project", "tickets"}

    for c in active_connections:
        c_id = c.get("connectorId", "").lower()
        meta = c.get("metadata", {}) or {}
        display = meta.get("displayName", c_id).lower()

        # Build dynamic token set from connectorId and displayName
        raw_tokens = {c_id, display, *display.split(), *c_id.split("_")}
        tokens = {tok for tok in raw_tokens if len(tok) >= 3 and tok not in generic_stopwords}

        if any(tok and tok in q for tok in tokens):
            matched.append(c)

    # Fail open: if no specific connector matched by name, query all active connections
    return matched if matched else active_connections


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERIC TOOL CLASSIFICATION (Discovery vs Action)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_tools(discovered_tools: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Classifies an app's tools into:
    - Discovery tools: 0 required arguments (e.g. list_organizations, whoami, getAccessibleResources, list_projects, conversations.list)
    - Action tools: has required arguments (e.g. search_issues, get_deployments, send_message)
    This dynamically guides the LLM to run discovery before guessing missing keys/IDs.
    """
    discovery, action = [], []
    for t in discovered_tools:
        required = (t.get("inputSchema", {}) or {}).get("required", [])
        if not required:
            discovery.append(t)
        else:
            action.append(t)
    return discovery, action


def _prune_tools_for_query(
    discovered_tools: List[Dict[str, Any]],
    user_query: str,
    connector_id: str = "",
    max_tools: int = 50,
) -> List[Dict[str, Any]]:
    """
    Guarantees tool schemas never exceed OpenAI's 128-tool limit.
    If a server exposes 100+ tools (e.g. Vercel with 212 tools, Atlassian with 200+ tools),
    we prioritize:
    1. All discovery tools (0 required args) so workspace IDs / org slugs are always discoverable.
    2. Action tools ranked by relevance to user query tokens + connector-specific relevance boost.
    """
    if len(discovered_tools) <= max_tools:
        return discovered_tools

    discovery, action = _classify_tools(discovered_tools)

    q_tokens = set(user_query.lower().replace(",", " ").replace("?", " ").split())
    q_tokens.update(["search", "list", "get", "find", "read", "fetch", "query", "status", "info"])

    def score_tool(tool: Dict[str, Any]) -> int:
        name = tool.get("name", "").lower()
        desc = (tool.get("description") or "").lower()
        score = 0
        # Boost tools matching connector ID (e.g. 'jira' in searchJiraIssuesUsingJql, 'sentry' in search_issues)
        if connector_id and connector_id.lower() in name:
            score += 5
        for tok in q_tokens:
            if len(tok) < 3:
                continue
            if tok in name:
                score += 3
            elif tok in desc:
                score += 1
        return score

    sorted_action = sorted(action, key=score_tool, reverse=True)
    action_budget = max(max_tools - len(discovery), 20)
    selected_action = sorted_action[:action_budget]

    return discovery + selected_action


# ─────────────────────────────────────────────────────────────────────────────
# 4. SINGLE ISOLATED APP WORKER (Dedicated Sub-Agent with Zero Tool Pollution)
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_single_app_worker(
    conn: Dict[str, Any],
    user_query: str,
    llm: Any,
    timeout_s: float = DEFAULT_WORKER_TIMEOUT_S,
    step_budget: int = DEFAULT_STEP_BUDGET,
) -> WorkerResult:
    """
    Runs an isolated sub-agent loop strictly for ONE connector.
    Bound only to that connector's tools, with timeout and error containment.
    """
    c_id = conn.get("connectorId", "").lower()
    token = conn.get("accessToken", "")
    meta = conn.get("metadata", {}) or {}
    mcp_url = meta.get("mcpUrl") or get_default_mcp_url(c_id)

    if not mcp_url or not token:
        return WorkerResult(connector=c_id, ok=False, error="Missing MCP URL or access token.")

    try:
        return await asyncio.wait_for(
            _run_worker_loop(c_id, mcp_url, token, meta, user_query, llm, step_budget),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[MCP WORKER] {c_id.capitalize()} worker timed out after {timeout_s}s.")
        return WorkerResult(connector=c_id, ok=False, error=f"Timed out after {timeout_s}s.")
    except Exception as e:
        logger.exception(f"[MCP WORKER] {c_id.capitalize()} worker crashed: {e}")
        return WorkerResult(connector=c_id, ok=False, error=str(e))


async def _run_worker_loop(
    c_id: str,
    mcp_url: str,
    token: str,
    meta: Dict[str, Any],
    user_query: str,
    llm: Any,
    step_budget: int,
) -> WorkerResult:
    """Core ReAct loop for a single connector with dynamic tool discovery and structured finalization."""
    discovered_tools = await fetch_mcp_tools_list_session(mcp_url, token)
    if not discovered_tools and meta.get("tools"):
        discovered_tools = [
            {"name": t, "description": f"Tool {t} on {c_id.capitalize()}", "inputSchema": {}}
            for t in meta.get("tools", [])
        ]

    if not discovered_tools:
        return WorkerResult(connector=c_id, ok=False, error=f"No accessible tools discovered on {c_id.capitalize()}.")

    # Prune tools to safe budget (guarantees <= 50 tools, well under OpenAI's 128 tool cap)
    active_tools = _prune_tools_for_query(discovered_tools, user_query, connector_id=c_id, max_tools=50)
    discovery_tools, _action_tools = _classify_tools(active_tools)

    tool_map: Dict[str, str] = {}
    openai_tools: List[Dict[str, Any]] = []

    for t in active_tools:
        raw_name = t.get("name", "")
        if not raw_name:
            continue
        fn_name = f"{c_id}__{raw_name}".replace("-", "_").replace(".", "_")
        base_fn_name, suffix = fn_name, 1
        while fn_name in tool_map:
            fn_name = f"{base_fn_name}_{suffix}"
            suffix += 1
        tool_map[fn_name] = raw_name
        openai_tools.append({
            "type": "function",
            "function": {
                "name": fn_name,
                "description": (t.get("description") or f"Call {raw_name} on {c_id.capitalize()}")[:1000],
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            },
        })

    # Structured Finalize Tool (Universal across all connectors)
    openai_tools.append({
        "type": "function",
        "function": {
            "name": FINALIZE_TOOL,
            "description": (
                "Call this tool to finalize your findings. Provide the retrieved items as a list of "
                "objects under 'data', any discovered workspace context (org slugs, project IDs, cloud IDs) "
                "under 'resolved_context', and a clear, factual markdown summary under 'summary'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Array of structured items (issues, errors, deployments, tasks, records) retrieved.",
                    },
                    "resolved_context": {
                        "type": "object",
                        "description": "Mapping of resolved context IDs (e.g. organization_slug, project_id, cloud_id).",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Concise, factual markdown summary of findings for this app.",
                    },
                },
                "required": ["data"],
            },
        },
    })

    llm_with_tools = llm.bind_tools(openai_tools)

    discovery_names = [t["name"] for t in discovery_tools] or ["none"]
    context_hints = []
    if meta.get("workspaceName"):
        context_hints.append(f"Workspace Name: '{meta['workspaceName']}'")
    if meta.get("organizationSlug") or meta.get("org") or meta.get("organization"):
        org_val = meta.get("organizationSlug") or meta.get("org") or meta.get("organization")
        context_hints.append(f"Known Organization Slug: '{org_val}'")
    if meta.get("projectSlug") or meta.get("projectName") or meta.get("projectId"):
        proj_val = meta.get("projectSlug") or meta.get("projectName") or meta.get("projectId")
        context_hints.append(f"Known Project / Slug: '{proj_val}'")
    if meta.get("cloudId"):
        context_hints.append(f"Known Cloud ID: '{meta['cloudId']}'")
    if meta.get("teamId"):
        context_hints.append(f"Known Team ID: '{meta['teamId']}'")
    if meta.get("region"):
        context_hints.append(f"Region: '{meta['region']}'")

    workspace_context_hint = " | ".join(context_hints) if context_hints else ""
    if workspace_context_hint:
        workspace_context_hint = f"Known Connection Metadata: {workspace_context_hint}."

    system_prompt = (
        f"You are the dedicated {c_id.capitalize()} Sub-Agent for WEKRAFT.\n"
        f"Task: Retrieve live third-party data to satisfy the user query: '{user_query}'.\n"
        f"{workspace_context_hint}\n"
        f"Tools with no required parameters are Discovery Tools: {discovery_names}.\n"
        f"Rules:\n"
        f"1. You MUST call retrieval tool(s) to fetch real workspace data. Do not provide a speculative text answer.\n"
        f"2. If an action tool requires an ID, slug, cloudId, channel, or resource key you do not have, "
        f"call a Discovery Tool first to discover it — NEVER guess or fabricate an ID.\n"
        f"3. When fetching lists of items, use reasonable general filters (e.g. limit: 20 or recent items) "
        f"so valid data is not filtered out by local display name mismatches.\n"
        f"4. If a call returns 404 or missing parameter, inspect the error, call a discovery tool if needed, and retry.\n"
        f"5. End your turn by calling `{FINALIZE_TOOL}` with the structured results ('data'), resolved context, and a clear markdown summary."
    )

    messages: List[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User Query: '{user_query}'"),
    ]

    tools_executed: List[str] = []
    seen_calls: Set[Tuple[str, Tuple]] = set()
    resolved_context: Dict[str, Any] = {}
    collected_tool_texts: List[str] = []
    final_payload: Optional[Dict[str, Any]] = None

    for step in range(step_budget):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # If model produced plain text instead of calling finalize_result, break and use text
            break

        for tc in tool_calls:
            fn_name = tc.get("name")
            args = tc.get("args") or {}
            call_id = tc.get("id", f"{c_id}_{step}")

            if fn_name == FINALIZE_TOOL:
                final_payload = args
                continue

            raw_tool_name = tool_map.get(fn_name)
            if not raw_tool_name:
                messages.append(ToolMessage(content=f"Error: Unknown tool '{fn_name}'.", tool_call_id=call_id))
                continue

            # Deduplication protection
            dedup_key = (raw_tool_name, tuple(sorted((k, str(v)) for k, v in args.items())))
            if dedup_key in seen_calls:
                messages.append(ToolMessage(
                    content="Tool was already called with these exact parameters. Use the prior output.",
                    tool_call_id=call_id,
                ))
                continue
            seen_calls.add(dedup_key)

            tools_executed.append(raw_tool_name)
            logger.info(f"[MCP AGENT] 🛠️ {c_id.capitalize()} executing '{raw_tool_name}' args={args}")

            # Emit live SSE event to frontend UI stream
            try:
                from langgraph.config import get_stream_writer
                writer = get_stream_writer()
                writer({"tool_called": f"{c_id.capitalize()}: {raw_tool_name}", "caller": "MCP Agent"})
            except Exception:
                pass

            res = await _call_tool_with_retry(mcp_url, token, raw_tool_name, args)
            content_str = str(res.get("content") or res.get("error") or "Empty result.")
            collected_tool_texts.append(f"[{raw_tool_name}]: {content_str}")

            # Truncate keeping head and tail to preserve essential IDs/cursors
            if len(content_str) > 6000:
                content_str = content_str[:3800] + "\n...[truncated]...\n" + content_str[-1500:]

            messages.append(ToolMessage(content=content_str, tool_call_id=call_id))

        if final_payload is not None:
            break

    # Extract structured results
    if final_payload is not None:
        resolved_context.update(final_payload.get("resolved_context", {}) or {})
        data = final_payload.get("data", []) or []
        if not isinstance(data, list):
            data = [data]
        summary = final_payload.get("summary")

        return WorkerResult(
            connector=c_id.capitalize(),
            ok=True,
            data=data,
            tools_executed=tools_executed,
            resolved_context=resolved_context,
            summary=summary,
        )

    # Fallback if finalize_result was not explicitly invoked
    if tools_executed:
        fallback_summary = "\n".join(collected_tool_texts)
        return WorkerResult(
            connector=c_id.capitalize(),
            ok=True,
            data=[{"raw_output": text} for text in collected_tool_texts],
            tools_executed=tools_executed,
            resolved_context=resolved_context,
            summary=fallback_summary[:2000],
        )

    return WorkerResult(
        connector=c_id.capitalize(),
        ok=False,
        error="No tool calls were executed.",
        tools_executed=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. MASTER DISPATCHER (Parallel Fan-Out & Synthesizer Formatting)
# ─────────────────────────────────────────────────────────────────────────────

async def execute_mcp_agent_workflow(
    project_id: str,
    user_query: str,
    user_name: Optional[str] = None,
    per_app_timeout_s: float = DEFAULT_WORKER_TIMEOUT_S,
) -> Dict[str, Any]:
    """
    Executes Generic Parallel MCP Sub-Agent Architecture:
    1. Discovers active project connections from Convex and routes via smart token matching.
    2. Spawns isolated, concurrent sub-agent workers per connected app via asyncio.gather().
    3. Aggregates structured data, tool logs, failure states, and zero-hallucination Markdown summary.
    """
    if not project_id:
        return {
            "summary": "⚠️ No active project specified for MCP integrations.",
            "structured_data": {},
            "connected_apps": [],
            "executed_tools": [],
            "failures": [],
        }

    connections = await fetch_project_mcp_connections_async(project_id)
    active_connections = [c for c in connections if c.get("accessToken")]

    if not active_connections:
        return {
            "summary": (
                "ℹ️ **No Connected MCP Integrations**: No third-party tools (Linear, Sentry, Jira, Vercel, Notion, Slack, Calendly) "
                "are currently connected to this workspace. Connect tools in the **Integrations** tab."
            ),
            "structured_data": {},
            "connected_apps": [],
            "executed_tools": [],
            "failures": [],
        }

    target_connections = smart_prune_connectors(user_query, active_connections)
    connected_app_names = [c.get("connectorId", "unknown").capitalize() for c in target_connections]

    mcp_model = os.getenv("MCP_AGENT_MODEL", "gpt-4.1-mini")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    llm = ChatOpenAI(
        model=mcp_model,
        openai_api_key=openai_api_key,
        temperature=0.0,
        max_retries=2,
    )

    t0 = time.monotonic()

    # FAN-OUT: Run all app sub-workers concurrently in parallel
    tasks = [
        _execute_single_app_worker(
            conn=conn,
            user_query=user_query,
            llm=llm,
            timeout_s=per_app_timeout_s,
        )
        for conn in target_connections
    ]

    results: List[WorkerResult] = await asyncio.gather(*tasks, return_exceptions=False)
    elapsed = time.monotonic() - t0

    # FAN-IN: Structured Aggregation
    structured_data: Dict[str, Any] = {}
    all_executed_tools: List[str] = []
    failures: List[Dict[str, str]] = []
    summary_sections: List[str] = []

    for res in results:
        all_executed_tools.extend(res.tools_executed)
        c_name = res.connector

        if not res.ok:
            err_msg = res.error or "Unknown worker error"
            failures.append({"connector": c_name, "error": err_msg})
            summary_sections.append(f"### {c_name} Status:\n⚠️ Could not retrieve data: {err_msg}")
            continue

        structured_data[c_name] = res.data
        app_blocks = []

        if res.summary:
            app_blocks.append(f"### {c_name} Summary:\n{res.summary.strip()}")

        if res.data:
            table_rows = [
                f"\n#### Live {c_name} Itemized Data ({len(res.data)} items found):",
            ]
            for idx, item in enumerate(res.data[:30], 1):
                if isinstance(item, dict):
                    title = item.get("title") or item.get("name") or item.get("summary") or item.get("id") or "Untitled"
                    short_id = item.get("shortId") or item.get("id") or item.get("key") or f"#{idx}"
                    culprit = item.get("culprit") or item.get("url") or item.get("projectName") or "-"
                    status = item.get("status") or item.get("state") or "-"
                    level = item.get("level") or item.get("priority") or "-"
                    count = item.get("events") or item.get("count") or "-"
                    users = item.get("users") or item.get("userCount") or "-"
                    link = item.get("permalink") or item.get("url") or item.get("inspectorUrl") or ""

                    link_md = f" | [Link]({link})" if link else ""
                    table_rows.append(
                        f"- **[{short_id}]** {title} | Location: `{culprit}` | Status: `{status}` | Level/Priority: `{level}` | Events: `{count}` | Users: `{users}`{link_md}"
                    )
                else:
                    table_rows.append(f"- {str(item)[:250]}")

            app_blocks.append("\n".join(table_rows))
        elif not res.summary:
            app_blocks.append(f"### {c_name} Findings:\nℹ️ Queried successfully; 0 items found matching this query.")

        summary_sections.append("\n\n".join(app_blocks))

    final_summary = "\n\n".join(summary_sections) if summary_sections else "No integration data retrieved."

    return {
        "summary": final_summary,
        "structured_data": structured_data,
        "connected_apps": connected_app_names,
        "executed_tools": all_executed_tools,
        "failures": failures,
        "elapsed_s": round(elapsed, 2),
    }
