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

DEFAULT_STEP_BUDGET = 5  # Raised from 3 — gives sub-agents enough room for discovery + action + finalize
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
    "github": "https://api.githubcopilot.com/mcp",
    "supabase": "https://mcp.supabase.com/mcp",
    "neon": "https://mcp.neon.tech/mcp",
    "stripe": "https://mcp.stripe.com",
    "posthog": "https://mcp.posthog.com/mcp",
}

APP_ALIASES: Dict[str, Set[str]] = {
    "jira": {"jira", "zira", "jirah", "jra", "atlassian", "jira-cloud", "jira-software", "jiracloud"},
    "linear": {"linear", "linar", "linera", "lineer", "linear-app"},
    "slack": {"slack", "slak", "slck", "slaack"},
    "calendly": {"calendly", "calendy", "calenderly", "calendli"},
    "notion": {"notion", "notn", "notio", "notion-app"},
    "sentry": {"sentry", "sentri", "sntery", "sentry-io"},
    "github": {"github", "gthub", "git-hub", "gh", "git"},
    "vercel": {"vercel", "vercl", "varcel"},
    "supabase": {"supabase", "superbase"},
    "neon": {"neon", "neondb"},
    "stripe": {"stripe"},
    "posthog": {"posthog", "post-hog"},
    "hubspot": {"hubspot", "hub-spot"},
    "mcp": {"mcp"},
}

# Connector category keywords for intent-based Smart Tool Pruning
CONNECTOR_KEYWORDS: Dict[str, List[str]] = {
    "slack": [
        "slack", "slak", "message", "channel", "chat", "team", "dm", "post", "send message", "thread", "conversation", "huddle"
    ],
    "calendly": [
        "calendly", "calendy", "schedule", "meeting", "booking", "availability", "slot", "invite", "calendar link", "event type", "reschedule"
    ],
    "linear": [
        "linear", "linar", "ticket", "issue", "cycle", "roadmap", "backlog item", "linear sprint", "linear task"
    ],
    "notion": [
        "notion", "doc", "page", "prd", "spec", "wiki", "workspace doc", "database", "notes", "rfc"
    ],
    "jira": [
        "jira", "zira", "jra", "atlassian", "epic", "story", "board", "jira ticket", "jira issue"
    ],
    "sentry": [
        "sentry", "sentri", "error", "exception", "crash", "stacktrace", "issue", "bug", "alert", "incident"
    ],
    "hubspot": [
        "hubspot", "crm", "contact", "deal", "lead", "company", "marketing", "pipeline", "sales"
    ],
    "vercel": [
        "vercel", "deployment", "deploy", "domain", "project", "env", "environment", "build", "alias", "preview"
    ],
    "github": [
        "github", "pull request", "pull requests", "github pr", "github repo", "github repository", "github branch", "github commit", "github release", "git merge", "gh"
    ],
    "supabase": [
        "supabase", "sql", "database", "table", "schema", "postgres", "edge function", "storage", "migration", "query", "row"
    ],
    "neon": [
        "neon", "database", "postgres", "branch", "serverless", "sql", "migration", "neon db", "connection string"
    ],
    "stripe": [
        "stripe", "payment", "invoice", "charge", "subscription", "customer", "payout", "refund", "billing", "card"
    ],
    "posthog": [
        "posthog", "analytics", "funnel", "event", "feature flag", "experiment", "trend", "user path", "cohort", "telemetry"
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
                try:
                    data = call_resp.json() if call_resp.text.strip() else {}
                except Exception:
                    data = {"raw_text": call_resp.text}

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

            if not data and call_resp.status_code == 202:
                return {"content": "Request accepted by MCP server.", "isError": False}

            return {"content": str(data.get("raw_text") or data), "isError": False}

        except httpx.HTTPStatusError as http_err:
            status = http_err.response.status_code
            if status in (401, 403):
                return {
                    "error": f"Authentication expired (HTTP {status} Unauthorized). Access token is expired or revoked. Please click 'Reconnect' in the Integrations tab.",
                    "isError": True,
                    "isAuthError": True,
                }
            return {"error": f"HTTP {status} error from MCP server: {http_err}", "isError": True}
        except Exception as err:
            err_msg = str(err)
            if "401" in err_msg or "unauthorized" in err_msg.lower():
                return {
                    "error": "Authentication expired (HTTP 401 Unauthorized). Access token is expired. Please click 'Reconnect' in the Integrations tab.",
                    "isError": True,
                    "isAuthError": True,
                }
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
            if res.get("isAuthError"):
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


def smart_prune_connectors(
    user_query: str,
    active_connections: List[Dict[str, Any]],
    active_skill_content: Optional[str] = None,
    selected_skill: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Matches query tokens or active skill against connectorId, displayName, and tokens.
    Returns matching connectors, or empty list if no third-party apps are targeted.
    Only queries all connectors if the user explicitly asks for general integrations/mcp.
    """
    if not active_connections:
        return []

    q = (user_query + " " + (selected_skill or "") + " " + (active_skill_content or "")).lower()
    matched = []
    generic_stopwords = {"workspace", "monitoring", "deployments", "integration", "app", "tools", "issue", "task", "project", "tickets"}

    for c in active_connections:
        c_id = c.get("connectorId", "").lower()
        meta = c.get("metadata", {}) or {}
        display = meta.get("displayName", c_id).lower()

        aliases = APP_ALIASES.get(c_id, {c_id})
        # Build dynamic token set from connectorId, aliases, and displayName
        raw_tokens = {c_id, display, *aliases, *display.split(), *c_id.split("_")}
        tokens = {tok for tok in raw_tokens if len(tok) >= 3 and tok not in generic_stopwords}

        if any(tok and tok in q for tok in tokens):
            matched.append(c)

    if matched:
        return matched

    # Explicit general MCP / integrations keywords check
    general_mcp_keywords = [
        "mcp", "integrations", "connected apps", "all apps", "external tools", "all tools", "third-party", "external integrations",
        # Cross-app discovery queries: user is comparing workspace tasks with external tools
        "compare", "bring", "import", "sync", "which tasks", "what tasks", "not completed", "incomplete", "missing from",
        "cross-check", "same as", "duplicate", "overlap", "suggest", "transfer", "migrate",
    ]
    if any(k in user_query.lower() for k in general_mcp_keywords):
        return active_connections

    return []


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
    max_tools: int = 20,
) -> List[Dict[str, Any]]:
    """
    Guarantees tool schemas never exceed compact token limits.
    Prioritizes discovery tools first, then query-relevant action tools.
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
    action_budget = max(max_tools - len(discovery), 10)
    selected_action = sorted_action[:action_budget]

    return discovery + selected_action


def _filter_tools_by_skill(
    discovered_tools: List[Dict[str, Any]],
    skill_content: str,
    connector_id: str = "",
) -> List[Dict[str, Any]]:
    """
    If a procedural skill is active, restrict discovered MCP tools STRICTLY to those mentioned in the skill.
    This prevents tool explosion (e.g. 44 Notion tools) and completely prevents the model from wandering
    into unwanted discovery/search tools (like notion-ai-search) when the skill dictates exact action tools.
    """
    if not skill_content or not discovered_tools:
        return discovered_tools

    skill_text = skill_content.lower()
    matched = []

    for tool in discovered_tools:
        raw_name = tool.get("name", "")
        if not raw_name:
            continue

        norm_name = raw_name.lower()
        norm_hyphen = norm_name.replace("_", "-")
        norm_underscore = norm_name.replace("-", "_")

        # Match exact name or hyphen/underscore variants in skill markdown
        if (
            f"`{norm_name}`" in skill_text
            or f"`{norm_hyphen}`" in skill_text
            or f"`{norm_underscore}`" in skill_text
            or f"tool: {norm_name}" in skill_text
            or f"tool: {norm_hyphen}" in skill_text
            or f"tool: {norm_underscore}" in skill_text
            or f"**tool:** {norm_name}" in skill_text
            or f"**tool:** {norm_hyphen}" in skill_text
            or f"**tool:** {norm_underscore}" in skill_text
            or norm_name in skill_text
            or norm_hyphen in skill_text
        ):
            matched.append(tool)

    return matched if matched else discovered_tools


# ─────────────────────────────────────────────────────────────────────────────
# 4. SKILL CONFIG EXTRACTOR (Pinned-Tool & Budget Parser)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_skill_config(
    skill_content: str,
    connector_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Reads the first ```json block inside a skill's markdown content and extracts:
      - pinnedTools[connector_id]: minimal tool schemas to use instead of live discovery
      - maxSteps: optional tighter step budget for this skill's workflows

    Returns None on any parse failure → caller falls back to full live discovery.
    This is intentionally lenient: a malformed JSON block or missing connector key
    silently degrades to explore mode rather than crashing.
    """
    import re
    if not skill_content or not connector_id:
        return None
    try:
        matches = re.findall(r'```json\s*(.*?)```', skill_content, re.DOTALL)
        for raw in matches:
            parsed = json.loads(raw.strip())
            pinned = parsed.get("pinnedTools", {})
            cid = connector_id.lower()
            if cid in pinned and isinstance(pinned[cid], list) and pinned[cid]:
                return {
                    "tools": pinned[cid],
                    "max_steps": parsed.get("maxSteps"),
                }
    except Exception:
        pass  # Any failure → graceful fallback to live discovery
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. SINGLE ISOLATED APP WORKER (Dedicated Sub-Agent with Zero Tool Pollution)
# ─────────────────────────────────────────────────────────────────────────────

async def _execute_single_app_worker(
    conn: Dict[str, Any],
    user_query: str,
    llm: Any,
    timeout_s: float = DEFAULT_WORKER_TIMEOUT_S,
    step_budget: int = DEFAULT_STEP_BUDGET,
    active_skill_content: Optional[str] = None,
    selected_skill: Optional[str] = None,
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
            _run_worker_loop(
                c_id, mcp_url, token, meta, user_query, llm, step_budget,
                active_skill_content=active_skill_content,
                selected_skill=selected_skill,
            ),
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
    active_skill_content: Optional[str] = None,
    selected_skill: Optional[str] = None,
) -> WorkerResult:
    """Core ReAct loop for a single connector with dynamic tool discovery and structured finalization."""

    # ── SMART DISCOVERY: Skill-driven mode vs Freeform Explore mode ──
    effective_budget = step_budget
    discovered_tools: List[Dict[str, Any]] = []

    skill_cfg = _extract_skill_config(active_skill_content or "", c_id) if active_skill_content else None
    if skill_cfg and skill_cfg.get("tools"):
        # Explicit pinned schemas in skill JSON
        discovered_tools = skill_cfg["tools"]
        if skill_cfg.get("max_steps"):
            effective_budget = min(int(skill_cfg["max_steps"]), step_budget)
        logger.info(f"[MCP AGENT] {c_id}: 📌 Pinned mode (schema) — {len(discovered_tools)} tools, budget={effective_budget}")
    else:
        # Live tool discovery from MCP server
        all_live_tools = await fetch_mcp_tools_list_session(mcp_url, token)
        if not all_live_tools and meta.get("tools"):
            all_live_tools = [
                {"name": t, "description": f"Tool {t} on {c_id.capitalize()}", "inputSchema": {}}
                for t in meta.get("tools", [])
            ]

        if active_skill_content:
            # Skill is active: strictly filter down to ONLY the tools defined in the skill!
            discovered_tools = _filter_tools_by_skill(all_live_tools, active_skill_content, c_id)
            logger.info(f"[MCP AGENT] {c_id}: 📌 Skill active ('{selected_skill}') — strictly filtered to {len(discovered_tools)} tools: {[t['name'] for t in discovered_tools]}")
        else:
            # No skill: Explore mode — freeform discovery
            discovered_tools = all_live_tools
            logger.info(f"[MCP AGENT] {c_id}: 🔍 Explore mode (no skill) — {len(discovered_tools)} tools discovered")
    # ──────────────────────────────────────────────────────────────────────────────────────────

    if not discovered_tools:
        return WorkerResult(connector=c_id, ok=False, error=f"No accessible tools discovered on {c_id.capitalize()}.")

    # If skill is active, preserve all skill tools; otherwise prune explore tools
    if active_skill_content:
        active_tools = discovered_tools
    else:
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
                        "description": "Array of structured items (issues, errors, deployments, tasks, records) retrieved or created.",
                    },
                    "resolved_context": {
                        "type": "object",
                        "description": "Mapping of resolved context IDs (e.g. organization_slug, project_id, cloud_id).",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Concise, factual markdown summary of findings or actions performed for this app.",
                    },
                },
                "required": ["data"],
            },
        },
    })

    llm_with_tools = llm.bind_tools(openai_tools, parallel_tool_calls=False)

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
    if meta.get("repoFullName") or meta.get("repoName"):
        repo_val = meta.get("repoFullName") or meta.get("repoName")
        context_hints.append(f"Connected GitHub Repository: '{repo_val}'")
    if meta.get("repoOwner"):
        context_hints.append(f"Repo Owner: '{meta['repoOwner']}'")
    if meta.get("repoId") or meta.get("repositoryId") or meta.get("githubId"):
        r_id = meta.get("repoId") or meta.get("repositoryId") or meta.get("githubId")
        context_hints.append(f"Repo ID: '{r_id}'")

    workspace_context_hint = f"Known Connection Metadata: {' | '.join(context_hints)}." if context_hints else ""

    github_scope_instruction = ""
    if c_id == "github":
        connected_repo_full = meta.get("repoFullName") or ""
        connected_repo_name = meta.get("repoName") or ""
        connected_repo_owner = meta.get("repoOwner") or ""
        connected_repo_id = meta.get("repoId") or meta.get("repositoryId") or meta.get("githubId") or ""
        connected_user = conn.get("userName") or meta.get("userName") or ""

        if not connected_repo_owner and "/" in connected_repo_full:
            connected_repo_owner = connected_repo_full.split("/")[0]
        if not connected_repo_name and "/" in connected_repo_full:
            connected_repo_name = connected_repo_full.split("/")[1]
        if not connected_repo_owner and connected_user:
            connected_repo_owner = connected_user

        target_repo_label = connected_repo_full or (f"{connected_repo_owner}/{connected_repo_name}" if connected_repo_owner and connected_repo_name else connected_repo_name)

        user_info = f"- GitHub Username: '{connected_user}'\n" if connected_user else ""

        if target_repo_label:
            github_scope_instruction = (
                f"\n\n🚨 STRICT SCOPE CONSTRAINT FOR GITHUB:\n"
                f"You are strictly authorized and scoped ONLY to the project's connected repository: '{target_repo_label}'.\n"
                f"{user_info}"
                f"- Repository Name: '{connected_repo_name}'\n"
                f"- Repository Owner: '{connected_repo_owner}'\n"
                f"- Repository ID / GitHub ID: '{connected_repo_id}'\n"
                f"RULES:\n"
                f"1. You MUST ONLY fetch, list, inspect, and summarize PRs, commits, branches, issues, and code for this specific connected repository ('{target_repo_label}').\n"
                f"2. NEVER query or return details for any other repository in GitHub.\n"
                f"3. Always pass owner='{connected_repo_owner}' and repo='{connected_repo_name or target_repo_label}' (or repo='{target_repo_label}') in all GitHub tool parameters.\n"
            )
        else:
            github_scope_instruction = (
                f"\n\n🚨 GITHUB REPOSITORY SCOPE NOTICE:\n"
                f"No specific repository is bound to this project in the database. Look for the repository matching project '{meta.get('projectName', '')}' or notify the user.\n"
            )

    jira_scope_instruction = ""
    if c_id == "jira":
        jira_scope_instruction = (
            f"\n\n🚨 JIRA QUERY RULES:\n"
            f"1. Never assume the Jira project key matches the WEKRAFT project name. Jira uses short project keys (e.g. 'KAN', 'PROJ', 'DEV').\n"
            f"2. By default, query tasks using JQL: `status IS NOT NULL ORDER BY updated DESC` (searches across all accessible projects in this Atlassian instance).\n"
            f"3. If the user explicitly mentions a Jira project name or key in their query (e.g. 'KAN', 'Payments'), use: `project = '<KeyOrName>' AND status IS NOT NULL ORDER BY updated DESC`. "
            f"If Jira returns an error that the project does not exist, immediately fall back to `status IS NOT NULL ORDER BY updated DESC`.\n"
        )

    skill_injection = ""
    if active_skill_content:
        skill_name_display = selected_skill or "Active Procedural Skill"
        skill_injection = (
            f"\n\n=== AUTHORITATIVE PROCEDURAL SKILL MANUAL: {skill_name_display} ===\n"
            f"{active_skill_content.strip()}\n"
            f"=========================================================================\n"
            f"INSTRUCTION: You MUST adhere strictly to the Step-by-Step Tool Execution Workflow and Anti-Hallucination rules defined in the Skill Manual above.\n"
        )

    if active_skill_content:
        system_prompt = (
            f"You are the dedicated {c_id.capitalize()} Sub-Agent executing a procedural skill workflow for WEKRAFT.\n"
            f"Target Task: {user_query}\n"
            f"{workspace_context_hint}\n"
            f"{github_scope_instruction}"
            f"{jira_scope_instruction}"
            f"{skill_injection}\n"
            f"CRITICAL EXECUTION RULES:\n"
            f"1. You are operating in STRICT SKILL EXECUTION MODE. Follow the exact step-by-step instructions in the Skill Manual above.\n"
            f"2. You are equipped ONLY with the exact tools needed for this skill: {[t['name'] for t in active_tools]}.\n"
            f"3. If the skill calls for creating or mutating resources (e.g. creating pages, issues, updating records), YOU MUST EXECUTE the corresponding creation/update tool. Do NOT skip tool calls or guess results.\n"
            f"4. NEVER output conversational filler like 'Please hold on', 'I will now compare', 'Please wait', or intermediate promises. You are a backend data worker.\n"
            f"5. Once tools return data, immediately extract the items (keys, titles, statuses, assignees) and call `{FINALIZE_TOOL}` with the structured output, URLs/IDs generated, and full markdown summary."
        )
    else:
        system_prompt = (
            f"You are the dedicated {c_id.capitalize()} Sub-Agent for WEKRAFT.\n"
            f"Task: Retrieve live third-party data to satisfy the user query: '{user_query}'.\n"
            f"{workspace_context_hint}\n"
            f"{github_scope_instruction}"
            f"{jira_scope_instruction}"
            f"Tools with no required parameters are Discovery Tools: {discovery_names}.\n"
            f"Rules:\n"
            f"1. You MUST call retrieval tool(s) to fetch real workspace data. Do not provide a speculative text answer.\n"
            f"2. If an action tool requires an ID, slug, cloudId, channel, or resource key you do not have, "
            f"call a Discovery Tool first to discover it — NEVER guess or fabricate an ID.\n"
            f"3. When fetching lists of items, use reasonable general filters (e.g. limit: 20 or recent items) "
            f"so valid data is not filtered out by local display name mismatches.\n"
            f"4. If a call returns 404 or missing parameter, inspect the error, call a discovery tool if needed, and retry.\n"
            f"5. NEVER output conversational filler like 'Please hold on', 'Please wait', or 'I will now compare'. End your turn by calling `{FINALIZE_TOOL}` with the structured results ('data'), resolved context, and a clear markdown summary of all retrieved items."
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

    last_text_response = ""
    for step in range(effective_budget):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)

        if getattr(response, "content", None):
            last_text_response = str(response.content).strip()

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
            # Raised limit: 5000 chars so LLM has richer context for cross-app discovery
            if len(content_str) > 5000:
                content_str = content_str[:3500] + "\n...[truncated]...\n" + content_str[-1500:]

            messages.append(ToolMessage(content=content_str, tool_call_id=call_id))

        if final_payload is not None:
            break

    # Helper to extract Notion URL & ID if Notion created a page
    import re
    import json
    notion_url_match = None
    notion_page_id = None
    if c_id == "notion":
        all_raw = "\n".join(collected_tool_texts) + " " + (last_text_response or "")

        # 1. Parse from JSON in tool execution outputs
        for text in collected_tool_texts:
            try:
                json_part = text
                if "]: " in text:
                    json_part = text.split("]: ", 1)[1]

                parsed = None
                try:
                    parsed = json.loads(json_part)
                except Exception:
                    pass

                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "text" in parsed[0]:
                    try:
                        parsed = json.loads(parsed[0]["text"])
                    except Exception:
                        pass

                if isinstance(parsed, dict):
                    pages = parsed.get("pages", [])
                    if pages and isinstance(pages, list) and isinstance(pages[0], dict):
                        if pages[0].get("url"):
                            notion_url_match = pages[0]["url"]
                        if pages[0].get("id"):
                            notion_page_id = pages[0]["id"]
            except Exception:
                pass

        # 2. Regex fallback
        if not notion_url_match:
            found_urls = re.findall(r'https?://(?:www\.)?notion\.(?:so|com)/[a-zA-Z0-9_\-]+', all_raw)
            if found_urls:
                notion_url_match = found_urls[0]

    # Helper function to sanitize summary and guarantee clean Notion markdown output
    def _sanitize_summary(raw_summary: str) -> str:
        s = raw_summary.strip()
        is_raw_json = s.startswith("{") and ("creation_mode" in s or "allow_async" in s or "pages" in s)
        if is_raw_json or (c_id == "notion" and "notion-create-pages" in tools_executed):
            link_md = f"[Open Notion Document]({notion_url_match})" if notion_url_match else "Notion Document"
            id_md = f"\n- **Page ID:** `{notion_page_id}`" if notion_page_id else ""
            return (
                f"✅ **Notion Page Created Successfully!**\n"
                f"- **Page Link:** {link_md}{id_md}\n"
                f"- **Status:** Published to Notion workspace.\n"
            )
        if notion_url_match and notion_url_match not in s:
            return f"✅ **Notion Page Created Successfully!**\n- **Page Link:** [Open Notion Document]({notion_url_match})\n\n" + s
        return s

    # Extract structured results
    if final_payload is not None:
        resolved_context.update(final_payload.get("resolved_context", {}) or {})
        data = final_payload.get("data", []) or []
        if not isinstance(data, list):
            data = [data]
        summary = _sanitize_summary(final_payload.get("summary") or last_text_response)

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
        fallback_summary = _sanitize_summary(last_text_response if last_text_response else "\n".join(collected_tool_texts))

        return WorkerResult(
            connector=c_id.capitalize(),
            ok=True,
            data=[{"raw_output": text} for text in collected_tool_texts],
            tools_executed=tools_executed,
            resolved_context=resolved_context,
            summary=fallback_summary[:3000],
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

def _emit_stream_status(
    status_text: Optional[str] = None,
    reasoning: Optional[str] = None,
    tool_called: Optional[str] = None,
    caller: Optional[str] = None,
):
    """Emits SSE custom event for frontend live reasoning and status."""
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        payload = {}
        if status_text:
            payload["agent_status"] = status_text
        if reasoning:
            payload["reasoning"] = reasoning
        if tool_called:
            payload["tool_called"] = tool_called
            if caller:
                payload["caller"] = caller
        if payload:
            writer(payload)
    except Exception:
        pass


async def execute_mcp_agent_workflow(
    project_id: str,
    user_query: str,
    user_name: Optional[str] = None,
    per_app_timeout_s: float = DEFAULT_WORKER_TIMEOUT_S,
    active_skill_content: Optional[str] = None,
    selected_skill: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes Generic Parallel MCP Sub-Agent Architecture:
    1. Discovers active project connections from Convex and routes via smart token matching.
    2. Spawns isolated, concurrent sub-agent workers per connected app via asyncio.gather().
    3. Ingests active procedural skill manual (if selected) into sub-agent execution.
    4. Aggregates structured data, tool logs, failure states, and zero-hallucination Markdown summary.
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

    target_connections = smart_prune_connectors(
        user_query,
        active_connections,
        active_skill_content=active_skill_content,
        selected_skill=selected_skill,
    )
    if not target_connections:
        return {
            "summary": "ℹ️ No third-party MCP integrations targeted for this request.",
            "structured_data": {},
            "connected_apps": [],
            "executed_tools": [],
            "failures": [],
        }
    connected_app_names = [c.get("connectorId", "unknown").capitalize() for c in target_connections]
    target_apps_str = ", ".join(connected_app_names) if connected_app_names else "workspace tools"

    if selected_skill:
        skill_clean = selected_skill.replace("_", " ")
        mcp_reasoning = f"Executing '{skill_clean}' skill: orchestrating tool queries across {target_apps_str} to extract real-time project context."
    else:
        mcp_reasoning = f"Querying connected workspace integrations ({target_apps_str}) to gather live issue tickets, incidents, and activity."

    _emit_stream_status(
        status_text=f"MCP Agent querying {target_apps_str}...",
        reasoning=mcp_reasoning,
    )

    mcp_model = os.getenv("MCP_AGENT_MODEL", "gpt-5-mini")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    kwargs: Dict[str, Any] = {
        "model": mcp_model,
        "openai_api_key": openai_api_key,
        "temperature": 0.0,
        "max_retries": 2,
    }
    if any(m in mcp_model.lower() for m in ["gpt-5", "o1", "o3", "o4"]):
        kwargs["model_kwargs"] = {"reasoning_effort": "low"}

    llm = ChatOpenAI(**kwargs)

    t0 = time.monotonic()

    # Two-Phase Coordination: If a destination/sink app (Notion) is targeted,
    # gather data from source apps or workspace DB first, then pass full context to sink.
    sink_connectors = {"notion"}
    source_conns = [c for c in target_connections if c.get("connectorId", "").lower() not in sink_connectors]
    sink_conns = [c for c in target_connections if c.get("connectorId", "").lower() in sink_connectors]

    results: List[WorkerResult] = []

    if sink_conns:
        # Phase 1: Run any third-party source workers in parallel
        source_results: List[WorkerResult] = []
        if source_conns:
            source_tasks = [
                _execute_single_app_worker(
                    conn=conn,
                    user_query=user_query,
                    llm=llm,
                    timeout_s=per_app_timeout_s,
                    active_skill_content=active_skill_content,
                    selected_skill=selected_skill,
                )
                for conn in source_conns
            ]
            source_results = await asyncio.gather(*source_tasks, return_exceptions=False)
            results.extend(source_results)

        # Gather internal workspace DB context (tasks, issues, workloads, project health)
        source_context_lines = []
        q_lower = (user_query + (active_skill_content or "")).lower()
        if project_id and any(k in q_lower for k in ["task", "issue", "workload", "health", "duration", "timeline", "project", "member", "internal", "wekraft"]):
            try:
                from app.agents.tools.tools import (
                    fetch_tasks_summary_async,
                    fetch_issues_summary_async,
                    fetch_member_workload_async,
                )
                tasks_f = fetch_tasks_summary_async(project_id)
                issues_f = fetch_issues_summary_async(project_id)
                workload_f = fetch_member_workload_async(project_id)
                t_res, i_res, w_res = await asyncio.gather(tasks_f, issues_f, workload_f, return_exceptions=True)

                if isinstance(t_res, dict) and not t_res.get("error"):
                    source_context_lines.append(f"### Internal Project Tasks:\n{t_res.get('summary') or str(t_res)}")
                if isinstance(i_res, dict) and not i_res.get("error"):
                    source_context_lines.append(f"### Internal Project Issues:\n{i_res.get('summary') or str(i_res)}")
                if isinstance(w_res, dict) and not w_res.get("error"):
                    members = w_res.get("members", [])
                    if members:
                        source_context_lines.append("### Team Member Workload:\n" + "\n".join(
                            f"- {m.get('name', 'Member')}: {m.get('activeTasksCount', 0)} active tasks, {m.get('totalLoggedHours', 0)} hrs logged"
                            for m in members
                        ))
            except Exception as e:
                logger.warning(f"Could not load internal project data for sink: {e}")

        for sr in source_results:
            if sr.ok and sr.summary:
                source_context_lines.append(f"### {sr.connector} Findings:\n{sr.summary}")
            elif sr.ok and sr.data:
                source_context_lines.append(f"### {sr.connector} Data ({len(sr.data)} items):\n" + "\n".join(str(d) for d in sr.data[:15]))

        aggregated_findings = "\n\n".join(source_context_lines) if source_context_lines else "No additional source items found."
        sink_query = (
            f"{user_query}\n\n"
            f"[RETRIEVED WORKSPACE DATA FROM SOURCES & DB]:\n"
            f"{aggregated_findings}\n\n"
            f"INSTRUCTION: Execute the active procedural skill to create/update the requested Notion document with the above retrieved data into organized markdown tables, including actionable insights and executive summary."
        )

        # Phase 2: Run sink sub-worker with gathered data
        sink_tasks = [
            _execute_single_app_worker(
                conn=conn,
                user_query=sink_query,
                llm=llm,
                timeout_s=per_app_timeout_s,
                active_skill_content=active_skill_content,
                selected_skill=selected_skill,
            )
            for conn in sink_conns
        ]
        sink_results: List[WorkerResult] = await asyncio.gather(*sink_tasks, return_exceptions=False)
        results.extend(sink_results)
    else:
        # Standard single-phase parallel execution
        tasks = [
            _execute_single_app_worker(
                conn=conn,
                user_query=user_query,
                llm=llm,
                timeout_s=per_app_timeout_s,
                active_skill_content=active_skill_content,
                selected_skill=selected_skill,
            )
            for conn in target_connections
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)
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
