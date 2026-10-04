"""
kaya_graph.py — Master Multi-Agent LangGraph for Kaya AI Product Manager.

Architecture:
1. Supervisor Router (Groq Fast Structured Intent Classifier)
2. Conditional Parallel Fan-Out:
   - 'direct_response' -> Kaya Direct Response (Groq 120b / fast streaming)
   - ['analyst', 'db_write', 'sprint'] -> Parallel Sub-Agent Worker execution
3. 3 Specialized Sub-Agents:
   - Analyst Sub-Agent (Parallel read tools: standups, tasks, issues, workloads, project timeline)
   - DB Write Sub-Agent (Task creation, issue creation, calendar events, report scheduler)
   - Sprint Sub-Agent (Sprint analytics/velocity, sprint creation, task allocation)
4. Fan-In Aggregation & Unified Synthesis:
   - Kaya Synthesizer (gpt-4.1-mini): Ingests all worker findings + project deadline awareness,
     streams final PM-level tokens to user.
5. Zero Cascading Failures:
   - Isolated try/except in every worker returning structured error envelopes to Kaya.
"""

import os
import re
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from langchain_core.runnables import RunnableConfig

from app.state.state import SupervisorState, RESET_SENTINEL
from app.core.utils.checkpointer import get_checkpointer
from app.agents.tools.tools import (
    fetch_user_standup_async,
    fetch_tasks_summary_async,
    fetch_issues_summary_async,
    fetch_member_workload_async,
    fetch_sprint_insights_async,
    fetch_active_skills_async,
    fetch_github_pull_requests_async,
    fetch_github_issues_async,
    fetch_github_contributor_activity_async,
    fetch_github_release_and_ci_status_async,
    write_calendar_event_to_convex,
    write_bulk_tasks_to_convex,
    write_bulk_issues_to_convex,
)
from app.agents.tools.mcp_client import (
    execute_mcp_agent_workflow,
    fetch_project_mcp_connections_async,
)
from app.core.utils.document_parser import get_parsed_document_from_cache

load_dotenv(override=True)



# ─────────────────────────────────────────────────────────────────────────────
# 1. SUPERVISOR ROUTER DECISION SCHEMA & PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    actions: List[str] = Field(
        description=(
            "List of sub-agent actions to trigger in parallel (select ONLY the sub-agents explicitly required):\n"
            "- 'github': Dedicated GitHub sub-agent for repository analysis: pull requests (stale PRs, reviews, blockers), GitHub issues, contributor velocity (commit trends), and GitHub release/CI build health.\n"
            "- 'mcp': ONLY for third-party external integrations (explicitly requested Jira, Linear, Slack, Calendly, Notion, Sentry, Vercel, MCP). NEVER call for internal project tasks/issues/sprints or if user hasn't asked for third party.\n"
            "- 'db_write': DB Write agent for creating tasks, issues, or calendar events in this project\n"
            "- 'analyst': Project Analyst agent for all internal read analytics: user daily standup, task summaries, issue tracking, member workloads, sprint insights/velocity, project health, deadlines\n"
            "- 'direct_response': simple greeting, casual conversation, or general product manager chat"
        )
    )
    selected_skill: Optional[str] = Field(
        default=None,
        description=(
            "Name of the SINGLE best matching skill from the Available Skills Catalog, "
            "or null if no specialized procedural skill applies."
        )
    )
    reasoning: str = Field(description="Clear reasoning for the sub-agent and skill routing choice.")


ROUTER_SYSTEM_PROMPT = """You are the Supervisor Router for the WEKRAFT AI Platform.
Analyze the user's incoming query (and previous query context if provided) and make two decisions:
1. Select the exact sub-agents required ('actions').
2. Select the SINGLE most appropriate procedural skill ('selected_skill') from the Available Skills Catalog, or null if none match.

CRITICAL RULES:
1. CREATING / IMPORTING TASKS, ISSUES, EVENTS IN THIS PROJECT:
   - When the user asks to create, add, insert, generate, or bring in tasks/issues (e.g., 'okay bring those task in, i want to create those task u said', 'create 3 issues for these errors', 'create tasks', 'schedule meeting', 'add those tasks into our project'), it targets this project's internal database -> route to 'db_write' (DB Write Agent) ONLY.
   - For follow-up task creation/importing commands ('bring those tasks in', 'create those tasks you said', 'add them to my project'): Select ONLY ['db_write'] and set selected_skill to null. NEVER re-invoke 'mcp', 'github', or 'analyst' for pure creation follow-ups.

2. GITHUB SUB-AGENT ('github'):
   - ONLY call 'github' when the user explicitly asks about repository code, pull requests (PRs, reviews, stale PRs), commits, developer velocity / contributor activity, or GitHub Actions CI/CD workflows and releases.
   - NEVER call 'github' for general internal project questions like "what are issues in my project", "what tasks do we have", "my standup", or "sprint status" unless they explicitly ask for GitHub/PR/commits.

3. THIRD-PARTY / MCP AGENT ('mcp'):
   - ONLY call 'mcp' if the user EXPLICITLY asks to query or mutate a connected third-party SaaS service (Jira, Linear, Slack, Calendly, Notion, Sentry, Vercel, HubSpot, Supabase, Stripe, PostHog, MCP).
   - PHONETIC & TYPO TOLERANCE: Treat common phonetic variations or typos of SaaS names as the intended integration (e.g., 'Zira'/'Jra'/'Atlassian' -> Jira, 'Linar' -> Linear, 'Slak' -> Slack).
   - NEGATION RESPECT: If the user says "without Jira", "don't check Sentry", "skip Slack", or "no external tools", do NOT include 'mcp'.

4. ANALYST AGENT ('analyst'):
   - Call 'analyst' for ALL internal project analytics: "what are issues in my project", task summaries, active bugs/issues in Convex, daily standups, member workload distribution, sprint progress & velocity, project health, deadlines, and PRD reviews.

5. MINIMAL SUB-AGENT CALLS:
   - Route ONLY to the exact sub-agents necessary. If it's a simple greeting or general PM question with no database access needed, route to 'direct_response'.

FEW-SHOT ROUTING EXAMPLES:
- Query: "what are issues in my project"
  -> actions: ["analyst"], selected_skill: null, reasoning: "User is asking for internal project issues in Convex database; route to analyst only."
- Query: "what are my issues"
  -> actions: ["analyst"], selected_skill: null, reasoning: "User is asking for their assigned internal work items; route to analyst only."
- Query: "What PRs are currently open or stale in our repository?"
  -> actions: ["github"], selected_skill: null, reasoning: "User is asking for repository pull request review status and stale PR analysis."
- Query: "Show me developer velocity and top contributors this week"
  -> actions: ["github"], selected_skill: null, reasoning: "User is requesting contributor commit velocity and team code activity."
- Query: "Did our latest CI workflow build pass on GitHub?"
  -> actions: ["github"], selected_skill: null, reasoning: "User is asking about GitHub Actions CI/CD workflow status."
- Query: "What tasks are blocked in our project?"
  -> actions: ["analyst"], selected_skill: null, reasoning: "Internal project task summary; route to analyst."
- Query: "Fetch the latest unresolved crashes from Sentry and list open Linear tickets"
  -> actions: ["mcp"], selected_skill: "incident_escalation_triage", reasoning: "External Sentry and Linear telemetry explicitly requested."
- Query: "Create 4 tasks for the Stripe webhook refactor"
  -> actions: ["db_write"], selected_skill: null, reasoning: "User wants to create internal project tasks."
- Query: "Hi Kaya, how are you doing today?"
  -> actions: ["direct_response"], selected_skill: null, reasoning: "Casual greeting requiring no tool or database execution."

Available Sub-Agents:
1. 'github': Dedicated GitHub agent for repository analytics (PRs, stale PRs, GitHub issues, contributor velocity, releases & CI/CD health).
2. 'mcp': Third-party SaaS integrations ONLY (Jira, Linear, Slack, Calendly, Notion, Sentry, HubSpot, Vercel).
3. 'db_write': DB Write agent for project mutations: internal task creation, issue creation, calendar event scheduling.
4. 'analyst': Project Analyst agent for all internal read-only analytics: daily standup, tasks, issues, workloads, sprint insights/velocity, project health, deadlines.
5. 'direct_response': Greetings, casual conversation, general PM advice without database queries.

Available Skills Catalog (Level 1 Metadata):
{SKILLS_CATALOG_BLOCK}

"""


# Canonical App Aliases mapping typos and variations to official connector IDs
APP_ALIASES: Dict[str, Set[str]] = {
    "jira": {"jira", "zira", "jirah", "jra", "atlassian", "jira-cloud", "jira-software", "jiracloud"},
    "linear": {"linear", "linar", "linera", "lineer", "linear-app"},
    "slack": {"slack", "slak", "slck", "slaack"},
    "calendly": {"calendly", "calendy", "calenderly", "calendli"},
    "notion": {"notion", "notn", "notio", "notion-app"},
    "sentry": {"sentry", "sentri", "sntery", "sentry-io"},
    "vercel": {"vercel", "vercl", "varcel"},
    "supabase": {"supabase", "superbase"},
    "neon": {"neon", "neondb"},
    "stripe": {"stripe"},
    "posthog": {"posthog", "post-hog"},
    "hubspot": {"hubspot", "hub-spot"},
    "mcp": {"mcp"},
}

STRICT_MCP_APP_NAMES = set(APP_ALIASES.keys())

def extract_mentioned_mcp_apps(text: str) -> Set[str]:
    """Extracts canonical MCP app names from text with typo and alias tolerance."""
    lower_text = text.lower()
    words = set(re.findall(r'[a-zA-Z0-9_\-]+', lower_text))
    matched_apps = set()
    for canonical_app, aliases in APP_ALIASES.items():
        if any(alias in words or f" {alias} " in f" {lower_text} " or f"'{alias}'" in lower_text or f'"{alias}"' in lower_text for alias in aliases):
            matched_apps.add(canonical_app)
    return matched_apps


def resolve_skill_selection(
    text: str,
    active_skills: List[Dict[str, Any]],
    selected_skill: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolves the best matching skill name and content.
    1. Verifies if selected_skill exists in active_skills.
    2. If missing, null, or mismatched, falls back to scoring active_skills against connector/tool keywords
       present in the query with alias/typo resolution.
       HEAVILY penalizes skills with 3rd-party connectors that the user DID NOT ask for.
    """
    if selected_skill:
        for s in active_skills:
            if s.get("name") == selected_skill:
                return selected_skill, s.get("content")

    mentioned_apps = extract_mentioned_mcp_apps(text)
    if not mentioned_apps:
        return None, None

    best_skill = None
    best_score = -100

    for s in active_skills:
        name = (s.get("name") or "").lower()
        connector = (s.get("connectorId") or s.get("connector") or "").lower()
        title = (s.get("title") or "").lower()
        desc = (s.get("description") or "").lower()
        connectors = [c.strip() for c in connector.split(",") if c.strip()]

        score = 0
        for app in mentioned_apps:
            if app in connectors:
                score += 20
            elif app in name:
                score += 15
            elif app in title or app in desc:
                score += 5

        for conn in connectors:
            if conn not in ["project", "internal"] and conn not in mentioned_apps:
                score -= 30

        if score > best_score and score > 0:
            best_score = score
            best_skill = s

    if best_skill:
        return best_skill.get("name"), best_skill.get("content")

    return None, None


def contains_exact_words(text: str, patterns: List[str]) -> bool:
    """Matches whole words or phrases in text using word boundaries (avoids 'pr' matching 'project')."""
    lower = text.lower()
    for item in patterns:
        if " " in item:
            if item in lower:
                return True
        else:
            if re.search(r'\b' + re.escape(item) + r'\b', lower):
                return True
    return False


NEGATION_PREFIXES = ("without", "don't", "dont", "do not", "skip", "ignore", "exclude", "no ", "never ")


def _sanitize_and_guard_decision(
    decision: SupervisorDecision,
    user_input: str,
    active_skills: List[Dict[str, Any]],
    has_attached_doc: bool = False,
) -> Tuple[SupervisorDecision, Optional[str]]:
    """
    Zero-Latency In-Memory Guard (< 0.05ms):
    - Respects negative qualifiers ("without Jira", "don't check Sentry").
    - Strips spurious 'mcp' calls if no real external SaaS app is requested.
    - Resolves procedural skill without network overhead.
    """
    latest_text = user_input
    if "Current Query:" in user_input:
        latest_text = user_input.split("Current Query:")[-1].strip()
    lower_latest = latest_text.lower()

    # 1. Ingest document hints if PRD/Doc is attached
    if has_attached_doc or "[attached:" in lower_latest or "prd" in lower_latest or "doc" in lower_latest:
        if any(k in lower_latest for k in ["create", "add", "make", "insert", "generate", "extract", "task", "tasks", "bulk"]):
            if "db_write" not in decision.actions:
                decision.actions.append("db_write")
            if "direct_response" in decision.actions:
                decision.actions.remove("direct_response")
        elif any(k in lower_latest for k in ["critical", "issue", "issues", "bug", "bugs", "risk", "risks", "review", "tell me", "what", "analyze", "explain"]):
            if "analyst" not in decision.actions:
                decision.actions.append("analyst")
            if "direct_response" in decision.actions:
                decision.actions.remove("direct_response")

    # 2. Check for explicit negations targeting SaaS apps (e.g. "without Jira", "skip sentry")
    negated_apps = set()
    for app in STRICT_MCP_APP_NAMES:
        aliases = APP_ALIASES.get(app, {app})
        for alias in aliases:
            if any(f"{neg}{alias}" in lower_latest or f"{neg} {alias}" in lower_latest for neg in NEGATION_PREFIXES):
                negated_apps.add(app)

    # 3. Detect positively requested SaaS apps (with typo and alias tolerance)
    mentioned_apps = extract_mentioned_mcp_apps(lower_latest) - negated_apps

    # 4. MCP Sanitization: If 'mcp' was added but no valid external app is mentioned (or all were negated)
    if "mcp" in decision.actions:
        if not mentioned_apps and not decision.selected_skill:
            decision.actions.remove("mcp")
            decision.reasoning += " (MCP removed: query contains no positive external SaaS target)"
            if not decision.actions:
                decision.actions = ["analyst"] if any(k in lower_latest for k in ["task", "tasks", "issue", "issues", "workload", "standup", "health", "project"]) else ["direct_response"]
    elif mentioned_apps and not any(neg in lower_latest for neg in ["without external", "no integrations"]):
        # Safe addition: user explicitly named an un-negated SaaS connector
        if "mcp" not in decision.actions:
            if "direct_response" in decision.actions:
                decision.actions.remove("direct_response")
            decision.actions.append("mcp")
            decision.reasoning += f" (MCP auto-included for {mentioned_apps})"

    # 5. Deterministic GitHub Keyword Safety Clamp (Strict Word Boundaries to avoid 'pr' in 'project')
    github_patterns = [
        "github", "pr", "prs", "pull request", "pull requests", "stale pr", "stale prs",
        "commit", "commits", "contributor", "contributors", "developer velocity",
        "ci status", "workflow run", "workflow runs", "build health", "github issue",
        "github issues", "github release", "git merge", "merge conflict", "codebase activity"
    ]
    is_github_intent = contains_exact_words(lower_latest, github_patterns)
    if is_github_intent and "github" not in decision.actions:
        if "direct_response" in decision.actions:
            decision.actions.remove("direct_response")
        decision.actions.append("github")

    # 6. Sprint & Analytics Keyword Safety Clamp (Skip if pure GitHub query)
    analytics_keywords = ["sprint", "sprints", "burndown", "task", "tasks", "standup", "workload", "health", "project", "timeline"]
    has_analyst_terms = any(k in lower_latest for k in analytics_keywords) or (("issue" in lower_latest or "issues" in lower_latest) and not is_github_intent)
    if has_analyst_terms and "analyst" not in decision.actions and "db_write" not in decision.actions:
        if "direct_response" in decision.actions:
            decision.actions.remove("direct_response")
        decision.actions.append("analyst")

    # 7. Clean up direct_response if actionable subagents exist
    if len(decision.actions) > 1 and "direct_response" in decision.actions:
        decision.actions.remove("direct_response")


    # Ensure actions is never empty
    if not decision.actions:
        decision.actions = ["direct_response"]

    # 7. Hydrate matched procedural skill
    matched_skill_name, matched_content = resolve_skill_selection(
        user_input, active_skills, decision.selected_skill
    )
    if matched_skill_name and not any(app in negated_apps for app in matched_skill_name.split("_")):
        decision.selected_skill = matched_skill_name
    else:
        decision.selected_skill = None
        matched_content = None

    return decision, matched_content


async def route_user_request(
    user_input: str,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    has_attached_doc: bool = False,
) -> Tuple[SupervisorDecision, Optional[str]]:
    """Evaluates user input using Groq 120b LLM router and applies zero-latency in-memory intent sanitization."""
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    router_model = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b")

    # Level 1: Fetch active skills catalog
    active_skills = await fetch_active_skills_async(user_id or "")
    catalog_lines = [
        f"- Skill: '{s.get('name', '')}' (Connectors: {s.get('connectorId', 'general')})\n  Description: {s.get('description', '')}"
        for s in active_skills
        if s.get("name")
    ]
    skills_catalog_block = "\n".join(catalog_lines) if catalog_lines else "No specialized skills registered."

    doc_hint = "\n[Notice: An uploaded PRD/specification document is attached in this session.]" if has_attached_doc else ""

    formatted_system_prompt = ROUTER_SYSTEM_PROMPT.replace("{SKILLS_CATALOG_BLOCK}", skills_catalog_block)

    messages = [
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(content=f"User Query Context:\n{user_input}{doc_hint}\n| Active Project ID: {project_id or 'none'}"),
    ]

    try:
        llm = ChatOpenAI(
            model=router_model,
            openai_api_key=groq_api_key,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.0,
            max_retries=2,
        ).with_structured_output(SupervisorDecision)

        raw_decision: SupervisorDecision = await llm.ainvoke(messages)
        if not raw_decision.actions:
            raw_decision.actions = ["direct_response"]

        # Apply zero-latency in-memory intent guard (< 0.05ms)
        decision, matched_content = _sanitize_and_guard_decision(
            raw_decision, user_input, active_skills, has_attached_doc=has_attached_doc
        )

        safe_reasoning = decision.reasoning.encode("ascii", "replace").decode("ascii")
        print(f"\n🎯 [ROUTER DECISION] Actions: {decision.actions} | Selected Skill: {decision.selected_skill or 'None'} | Reasoning: {safe_reasoning}\n")
        return decision, matched_content

    except Exception as e:
        safe_err = str(e).encode("ascii", "replace").decode("ascii")
        print(f"[ROUTER WARNING] Fallback routing due to: {safe_err}")
        actions = []
        latest_text = user_input
        if "Current Query:" in user_input:
            latest_text = user_input.split("Current Query:")[-1].strip()
        lower_latest = latest_text.lower()

        if any(k in lower_latest for k in ["create", "add", "make", "insert", "schedule", "extract"]) and any(k in lower_latest for k in ["task", "tasks", "issue", "issues", "event", "these", "those"]):
            actions.append("db_write")
        elif any(k in lower_latest for k in ["task", "tasks", "issue", "issues", "standup", "workload", "health", "project", "critical", "prd", "doc", "sprint", "sprints", "velocity"]):
            actions.append("analyst")
        if any(k in lower_latest for k in STRICT_MCP_APP_NAMES) and not any(f"{neg}{k}" in lower_latest or f"{neg} {k}" in lower_latest for neg in NEGATION_PREFIXES for k in STRICT_MCP_APP_NAMES):
            actions.append("mcp")
        if not actions:
            actions = ["direct_response"]

        raw_fallback = SupervisorDecision(
            actions=actions,
            selected_skill=None,
            reasoning=f"Fallback routing: {safe_err}",
        )
        return _sanitize_and_guard_decision(raw_fallback, user_input, active_skills, has_attached_doc=has_attached_doc)



# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _get_latest_user_text(messages: List[BaseMessage]) -> str:
    """Extracts text content of the latest human message."""
    for m in reversed(messages):
        if getattr(m, "type", None) == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
        elif isinstance(m, dict):
            role = m.get("role") or m.get("type")
            if role in ["human", "user"]:
                return m.get("content", "")
    return ""


def _get_router_user_query(messages: List[BaseMessage]) -> str:
    """
    Extracts current query (current) + previous user query (current - 1)
    to provide the Supervisor Router with conversational context for follow-up questions.
    """
    human_texts: List[str] = []
    for m in messages:
        if getattr(m, "type", None) == "human":
            text = m.content if isinstance(m.content, str) else str(m.content)
            if text and text.strip():
                human_texts.append(text.strip())
        elif isinstance(m, dict):
            role = m.get("role") or m.get("type")
            if role in ["human", "user"]:
                text = m.get("content", "")
                if isinstance(text, str) and text.strip():
                    human_texts.append(text.strip())

    if not human_texts:
        return ""
    if len(human_texts) == 1:
        return f"Current Query: {human_texts[-1]}"

    current_query = human_texts[-1]
    prev_query = human_texts[-2]
    return f"Previous Query: {prev_query}\nCurrent Query: {current_query}"


def _emit_stream_status(
    status_text: Optional[str] = None,
    reasoning: Optional[str] = None,
    subagent_called: Optional[str] = None,
    tool_called: Optional[str] = None,
    caller: Optional[str] = None,
    skill_activated: Optional[str] = None,
):
    """Emits custom SSE progress event for frontend status indicator."""
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        payload = {}
        if status_text:
            payload["agent_status"] = status_text
        if reasoning:
            payload["reasoning"] = reasoning
        if subagent_called:
            payload["subagent_called"] = subagent_called
        if tool_called:
            payload["tool_called"] = tool_called
            if caller:
                payload["caller"] = caller
        if skill_activated:
            payload["skill_activated"] = skill_activated
        if payload:
            writer(payload)
    except Exception:
        pass


def get_analyst_llm() -> ChatOpenAI:
    """Returns model for Analyst worker reasoning and document evaluation (gpt-5-mini with low reasoning effort for minimal latency)."""
    analyst_model = os.getenv("ANALYST_MODEL", "gpt-5-mini")
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        kwargs: Dict[str, Any] = {
            "model": analyst_model,
            "openai_api_key": openai_key,
            "temperature": 0.1,
        }
        if any(m in analyst_model.lower() for m in ["gpt-5", "o1", "o3", "o4"]):
            kwargs["model_kwargs"] = {"reasoning_effort": "low"}
        return ChatOpenAI(**kwargs)
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatOpenAI(
        model=os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b"),
        openai_api_key=groq_key or "none",
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.1,
    )


def get_github_llm() -> ChatOpenAI:

    """Returns model for GitHub Sub-Agent worker (gpt-4.1-mini)."""
    github_model = os.getenv("GITHUB_MODEL", "gpt-4.1-mini")
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return ChatOpenAI(
            model=github_model,
            openai_api_key=openai_key,
            temperature=0.1,
        )
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatOpenAI(
        model=os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b"),
        openai_api_key=groq_key or "none",
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────

async def supervisor_router_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Evaluates user query and determines parallel execution branches."""
    messages = state.get("messages", [])
    user_query = _get_router_user_query(messages)
    project_id = state.get("project_id")
    user_id = state.get("user_id")
    file_id = state.get("file_id")

    _emit_stream_status(status_text="Kaya is thinking...")
    decision, matched_content = await route_user_request(
        user_query,
        project_id=project_id,
        user_id=user_id,
        has_attached_doc=bool(file_id),
    )

    # Immediately emit reasoning event so frontend displays it right after thinking
    _emit_stream_status(status_text="Kaya is reasoning...", reasoning=decision.reasoning)

    if decision.selected_skill:
        _emit_stream_status(
            status_text=f"Loaded procedural skill: '{decision.selected_skill}'",
            skill_activated=decision.selected_skill,
        )

    # Emit subagent delegation events for actions selected
    for action in decision.actions:
        if action == "analyst":
            _emit_stream_status(subagent_called="analyst_agent")
        elif action == "db_write":
            _emit_stream_status(subagent_called="db_write_agent")
        elif action == "mcp":
            _emit_stream_status(subagent_called="mcp_agent")
        elif action == "github":
            _emit_stream_status(subagent_called="github_agent")

    return {
        "next": decision.actions,
        "router_reasoning": decision.reasoning,
        "action_type": decision.actions[0] if len(decision.actions) == 1 else "multi_agent",
        "selected_skill": decision.selected_skill,
        "active_skill_content": matched_content,
    }


def route_supervisor(state: SupervisorState) -> List[str]:
    """Conditional fan-out edge: maps decision actions to node names."""
    actions = state.get("next", ["direct_response"])
    if isinstance(actions, str):
        actions = [actions]

    if not actions or "direct_response" in actions:
        return ["kaya_direct_node"]

    target_nodes = []
    if "analyst" in actions:
        target_nodes.append("analyst_node")
    if "db_write" in actions:
        target_nodes.append("db_write_node")
    if "mcp" in actions:
        target_nodes.append("mcp_node")
    if "github" in actions:
        target_nodes.append("github_node")

    return target_nodes if target_nodes else ["kaya_direct_node"]


# ─────────────────────────────────────────────────────────────────────────────
# SUB-AGENT 1: ANALYST WORKER (Parallel Read Tools + Sprint Insights)
# ─────────────────────────────────────────────────────────────────────────────

async def analyst_worker_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Analyst Sub-Agent (Powered by gpt-5-mini):
    - Runs tools in PARALLEL via asyncio.gather (standup, tasks, issues, workloads, sprints).
    - Queries sprint insights specifically when sprint/velocity/milestone context is requested.
    - Normalizes results into non-empty structured Markdown findings.
    - Wrapped in try/except for graceful error reporting.
    """
    _emit_stream_status(status_text="Analyst worker analyzing project health, tasks & sprint data...")
    project_id = state.get("project_id") or ""
    user_id = state.get("user_id") or ""
    messages = state.get("messages", [])
    user_query = _get_latest_user_text(messages).lower()

    if not project_id:
        return {
            "_analyst_messages": [
                RESET_SENTINEL,
                {"role": "analyst", "content": "⚠️ No active project selected. Cannot fetch analytics."},
            ]
        }

    try:
        # Determine specific tools needed based on user query
        has_task_kw = any(k in user_query for k in ["task", "tasks", "todo", "to-do", "to do", "assigned", "backlog", "story", "stories"])
        has_issue_kw = any(k in user_query for k in ["issue", "issues", "bug", "bugs", "error", "errors", "crash", "sentry", "linear", "defect", "incident", "problem", "problems", "fail", "failure", "unresolved"])
        has_standup_kw = any(k in user_query for k in [
            "standup", "stand up", "my work", "my task", "my tasks", "my issues",
            "what is my", "what i need to do", "what should i do", "today's work",
            "what did i do", "assigned to me", "my assignment", "my work items",
            "what am i working on", "my status", "for me", "me"
        ])
        has_workload_kw = any(k in user_query for k in ["workload", "team", "members", "who is working", "capacity", "distribution"])
        has_sprint_kw = any(k in user_query for k in ["sprint", "sprints", "velocity", "burndown", "active sprint", "current sprint", "milestone", "milestones"])
        is_general_query = not (has_task_kw or has_issue_kw or has_standup_kw or has_workload_kw or has_sprint_kw)

        need_tasks = has_task_kw or is_general_query
        need_issues = has_issue_kw or is_general_query
        need_standup = has_standup_kw or is_general_query
        need_workload = has_workload_kw
        need_sprint = has_sprint_kw or is_general_query

        # Formulate active Project Analyst reasoning
        reasoning_actions = []
        if need_standup:
            reasoning_actions.append("your personal standup and assigned work items")
        if need_tasks:
            reasoning_actions.append("active project task backlog")
        if need_issues:
            reasoning_actions.append("unresolved issues & blocker bugs")
        if need_workload:
            reasoning_actions.append("team member workload distribution")
        if need_sprint:
            reasoning_actions.append("sprint velocity and timeline progress")
        if state.get("file_id"):
            reasoning_actions.append("attached PRD/spec document architecture")

        reasoning_summary = ", ".join(reasoning_actions) if reasoning_actions else "project tasks, issues, and sprint status"
        analyst_reasoning = f"Analyzing {reasoning_summary} to evaluate progress, team workload, and critical path blockers."

        _emit_stream_status(
            status_text="Project Analyst is analyzing...",
            reasoning=analyst_reasoning,
        )

        if need_tasks:
            _emit_stream_status(tool_called="get_tasks_summary", caller="Project analyst")
        if need_issues:
            _emit_stream_status(tool_called="get_issues_summary", caller="Project analyst")
        if need_standup:
            _emit_stream_status(tool_called="get_user_standup", caller="Project analyst")
        if need_workload:
            _emit_stream_status(tool_called="get_member_workload", caller="Project analyst")
        if need_sprint:
            _emit_stream_status(tool_called="get_sprint_insights", caller="Project analyst")

        # Execute read tools in parallel concurrently (only those actually needed)
        tasks_future = fetch_tasks_summary_async(project_id) if need_tasks else asyncio.sleep(0, result=None)
        issues_future = fetch_issues_summary_async(project_id) if need_issues else asyncio.sleep(0, result=None)
        standup_future = fetch_user_standup_async(project_id, user_id) if (need_standup and user_id) else asyncio.sleep(0, result=None)
        workload_future = fetch_member_workload_async(project_id) if need_workload else asyncio.sleep(0, result=None)
        sprint_future = fetch_sprint_insights_async(project_id) if need_sprint else asyncio.sleep(0, result=None)

        tasks_res, issues_res, standup_res, workload_res, sprint_res = await asyncio.gather(
            tasks_future, issues_future, standup_future, workload_future, sprint_future, return_exceptions=True
        )

        # Normalize exceptions if any individual tool failed
        tasks_data = tasks_res if not isinstance(tasks_res, Exception) else {"error": str(tasks_res)}
        issues_data = issues_res if not isinstance(issues_res, Exception) else {"error": str(issues_res)}
        standup_data = standup_res if not isinstance(standup_res, Exception) else {"error": str(standup_res)}
        workload_data = workload_res if not isinstance(workload_res, Exception) else {"error": str(workload_res)}
        sprint_data = sprint_res if not isinstance(sprint_res, Exception) else {"error": str(sprint_res)}

        # Build clean structured summary for Kaya Synthesis
        lines = ["### Analyst Sub-Agent Findings:"]

        # Tasks Section (only if tasks were requested)
        if tasks_data is not None:
            if "error" in tasks_data:
                lines.append(f"- **Tasks Summary**: ⚠️ {tasks_data['error']}")
            else:
                total = tasks_data.get("totalCount", tasks_data.get("total", 0))
                completed = tasks_data.get("completedCount", 0)
                blocked = tasks_data.get("blockedCount", 0)
                active_tasks = tasks_data.get("tasks") or tasks_data.get("criticalAndActiveTasks") or []
                lines.append(f"- **Tasks Breakdown**: Total Tasks: {total} | Completed: {completed} | Blocked: {blocked}")
                if active_tasks:
                    lines.append("  **Active Tasks List**:")
                    for t in active_tasks:
                        if isinstance(t, str):
                            lines.append(f"  • {t}")
                        elif isinstance(t, dict):
                            title = t.get("title", "Untitled")
                            prio = t.get("priority", "medium")
                            assignee = t.get("assignee") or (", ".join(t.get("assignees", [])) if isinstance(t.get("assignees"), list) else "Unassigned")
                            deadline_status = t.get("deadlineStatus", "on track")
                            lines.append(f"  • Task Title: '{title}' | Assignee: {assignee} | Priority: {prio} | Deadline: {deadline_status}")

        # Issues Section (only if issues were requested)
        if issues_data is not None:
            if "error" in issues_data:
                lines.append(f"- **Issues Summary**: ⚠️ {issues_data['error']}")
            else:
                total_issues = issues_data.get("totalCount", issues_data.get("total", 0))
                critical = issues_data.get("criticalCount", 0)
                active_issues = issues_data.get("activeIssues", [])
                lines.append(f"- **Active Issues Breakdown**: Total Issues: {total_issues} | Critical Blockers: {critical}")
                if active_issues:
                    lines.append("  **Active Issues List**:")
                    for iss in active_issues:
                        title = iss.get("title", "Untitled")
                        sev = iss.get("severity", "medium")
                        status = iss.get("status", "opened")
                        assignees = ", ".join(iss.get("assignees", [])) if isinstance(iss.get("assignees"), list) else "Unassigned"
                        lines.append(f"  • Issue Title: '{title}' | Severity: {sev} | Status: {status} | Assignee: {assignees}")

        # Sprint Insights Section (only when sprint data was requested)
        if need_sprint and sprint_data is not None:
            if "error" in sprint_data:
                lines.append(f"- **Sprint Insights**: ⚠️ {sprint_data['error']}")
            else:
                sprints_list = sprint_data.get("sprints", [])
                active_sprints = [s for s in sprints_list if s.get("status") == "active"]
                planned_sprints = [s for s in sprints_list if s.get("status") == "planned"]
                completed_sprints = [s for s in sprints_list if s.get("status") == "completed"]
                lines.append(
                    f"- **Sprint Analytics**: Total Sprints: {len(sprints_list)} | "
                    f"Active: {len(active_sprints)} | Planned (Not Started): {len(planned_sprints)} | Completed: {len(completed_sprints)}"
                )
                for s in sprints_list:
                    s_name = s.get("name", "Sprint")
                    s_status = s.get("status", "planned")
                    s_goal = s.get("goal") or "No goal specified"
                    stats = s.get("stats", {})
                    progress_pct = stats.get("progressPercent", 0)
                    duration = s.get("duration", {})
                    dur_str = f" ({duration.get('start', '')} - {duration.get('end', '')})" if duration else ""
                    lines.append(
                        f"  • Sprint '{s_name}' [{s_status.upper()}]{dur_str} | Goal: {s_goal} | "
                        f"Progress: {progress_pct}% (Completed Tasks: {stats.get('completedTasks', 0)}/{stats.get('totalTasks', 0)}, "
                        f"Closed Issues: {stats.get('closedIssues', 0)}/{stats.get('totalIssues', 0)})"
                    )

        # Standup Section
        if need_standup and standup_data and "error" not in standup_data:
            user_tasks = standup_data.get("tasks", []) or standup_data.get("assignedTasks", [])
            lines.append(f"- **Current User Work & Standup**: {len(user_tasks)} task(s) directly assigned to you.")
            for ut in user_tasks:
                if isinstance(ut, dict):
                    lines.append(f"  • Task: '{ut.get('title', 'Task')}' (Status: {ut.get('status', 'unknown')}, Priority: {ut.get('priority', 'medium')})")
                else:
                    lines.append(f"  • Task: '{ut}'")

        # Workload Section
        if need_workload and workload_data and "error" not in workload_data:
            members = workload_data.get("members", [])
            lines.append(f"- **Team Workload Breakdown**: {len(members)} team member(s) tracked.")
            for m in members:
                m_name = m.get("name", "Unknown")
                m_role = m.get("role", "member")
                tasks_val = m.get("activeTasksCount") if m.get("activeTasksCount") is not None else (m.get("totalTasks") or 0)
                issues_val = m.get("activeIssuesCount") if m.get("activeIssuesCount") is not None else (m.get("totalIssues") or 0)
                status_note = m.get("statusNote") or m.get("workloadStatus", "balanced")
                blocked_val = m.get("blockedCount", 0)
                overdue_val = m.get("overdueCount", 0)
                
                status_indicator = "⚖️"
                if m.get("workloadStatus") == "overloaded":
                    status_indicator = "🔥 [OVERLOADED]"
                elif m.get("workloadStatus") == "stale_or_blocked":
                    status_indicator = "⚠️ [STALE / BLOCKED]"
                elif m.get("workloadStatus") == "idle":
                    status_indicator = "💤 [IDLE]"
                else:
                    status_indicator = "✅ [BALANCED]"
                    
                lines.append(
                    f"  • {status_indicator} **{m_name}** ({m_role}): Active Tasks: {tasks_val} | Active Issues: {issues_val} "
                    f"| Blocked: {blocked_val} | Overdue: {overdue_val} — Status: _{status_note}_"
                )

        # Document Analysis Section (if PRD/specification doc is attached or referenced)
        file_id = state.get("file_id")
        if file_id and user_id:
            try:
                _emit_stream_status(tool_called="parse_document_eval", caller="Project analyst")
                doc_data = await get_parsed_document_from_cache(user_id=user_id, file_id=file_id)
                if doc_data and doc_data.get("parsed_markdown"):
                    filename = doc_data.get("file_name", "Uploaded PRD/Document")
                    doc_md = doc_data.get("parsed_markdown", "")
                    
                    eval_prompt = [
                        SystemMessage(content=(
                            "You are a Senior Technical Product Manager & Systems Architect on WEKRAFT.\n"
                            "Analyze the attached PRD / specification document and produce an executive PM breakdown:\n"
                            "1. Document Scope & Core Objectives: Short 2-sentence summary.\n"
                            "2. Top Critical Issues, Blockers & System Bottlenecks: Highlight highest risk technical flaws, unhandled error conditions, or architectural risks (ranked by severity: Critical, High, Medium).\n"
                            "3. Gaps, Missing Specs & Security/Compliance Concerns: Edge cases or unaddressed requirements.\n"
                            "4. Recommended Action Items: Concrete next steps for the engineering team.\n"
                            "Be crisp, highly specific, reference real requirements from the document, and format with clear markdown bullet points and mini tables."
                        )),
                        HumanMessage(content=f"Document Filename: {filename}\n\nDocument Content:\n{doc_md[:25000]}\n\nUser Question/Focus: '{user_query}'"),
                    ]
                    _emit_stream_status(
                        status_text=f"Project Analyst evaluating {filename}...",
                        reasoning=f"Evaluating technical specifications in '{filename}' to identify architectural risks, unhandled edge cases, and PM action items.",
                    )
                    eval_llm = get_analyst_llm()
                    eval_res = await eval_llm.ainvoke(eval_prompt)
                    lines.append(f"\n### PRD / Document Review & Critical Issues ({filename}):\n{eval_res.content}")
            except Exception as doc_err:
                print(f"[ANALYST WORKER] Document analysis notice: {doc_err}")
                lines.append(f"- **Document Analysis Notice**: {doc_err}")

        findings_text = "\n".join(lines)

        return {
            "_analyst_messages": [RESET_SENTINEL, {"role": "analyst", "content": findings_text}],
            "standup_data": standup_data if need_standup else None,
            "sprint_insights": sprint_data if need_sprint else None,
            "file_id": file_id,
        }

    except GraphInterrupt:
        raise
    except Exception as e:
        error_msg = f"Analyst Sub-Agent encountered an error: {e}"
        print(f"[ANALYST WORKER ERROR] {error_msg}")
        return {
            "_analyst_messages": [
                RESET_SENTINEL,
                {"role": "analyst", "content": f"⚠️ [Analyst Data Notice]: {error_msg}"},
            ],
            "active_error": error_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-AGENT 2: DB WRITE WORKER (Mem0 & HITL Write Schema Tools)
# ─────────────────────────────────────────────────────────────────────────────

def _get_tag_color(tag_label: str) -> str:
    """Map a tag label to one of the theme badge colors: green, yellow, purple, blue, grey."""
    tl = (tag_label or "").lower().strip()
    if any(k in tl for k in ["pay", "bill", "money", "financ", "subscri", "pricing"]):
        return "green"
    elif any(k in tl for k in ["middle", "api", "back", "rout", "serv", "data", "db", "sql", "endpoint"]):
        return "yellow"
    elif any(k in tl for k in ["ui", "front", "design", "client", "meet", "ux", "mobile", "view", "page"]):
        return "purple"
    elif any(k in tl for k in ["bug", "fix", "issue", "err", "test", "doc", "audit", "patch"]):
        return "grey"
    elif any(k in tl for k in ["auth", "sec", "iam", "devops", "cloud", "infra", "setup", "config"]):
        return "blue"
    colors = ["blue", "purple", "green", "yellow", "grey"]
    return colors[sum(ord(c) for c in tl) % len(colors)]


class TaskDraftItem(BaseModel):
    title: str = Field(description="Clean, concise title of the task")
    description: Optional[str] = Field(default="", description="Detailed description or context")
    priority: Optional[str] = Field(default="medium", description="Priority level: 'high', 'medium', or 'low'")
    tag: Optional[str] = Field(
        default="Feature",
        description="A short, concise 1-2 word domain tag describing the task category (e.g. 'Payment', 'Auth', 'Middleware', 'UI', 'Backend', 'DevOps', 'Client', 'Setup', 'Bugfix', 'API', 'Billing', 'Testing'). Keep it very short. NEVER use 'PRD-Import'."
    )


class IssueDraftItem(BaseModel):
    title: str = Field(description="Clean, concise title of the issue or bug")
    description: Optional[str] = Field(default="", description="Detailed description of bug or issue")
    severity: Optional[str] = Field(default="medium", description="Severity level: 'critical', 'high', 'medium', or 'low'")
    priority: Optional[str] = Field(default="medium", description="Priority: 'high', 'medium', 'low'")


class CalendarDraftEvent(BaseModel):
    title: str = Field(description="Title of meeting or event")
    description: Optional[str] = Field(default="", description="Event description or agenda")
    event_type: Optional[str] = Field(default="event", description="'event' or 'milestone'")
    start_iso: Optional[str] = Field(default="", description="Start time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)")
    end_iso: Optional[str] = Field(default="", description="End time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)")
    all_day: Optional[bool] = Field(default=False, description="Whether event is all day")


class DBWriteIntentExtraction(BaseModel):
    is_task_creation: bool = Field(default=False, description="True if user wants to create one or more tasks")
    tasks: List[TaskDraftItem] = Field(default_factory=list, description="List of tasks to create")
    is_issue_creation: bool = Field(default=False, description="True if user wants to create one or more issues")
    issues: List[IssueDraftItem] = Field(default_factory=list, description="List of issues to create")
    is_calendar_event: bool = Field(default=False, description="True if user wants to schedule a calendar event or meeting")
    calendar_event: Optional[CalendarDraftEvent] = Field(default=None, description="Calendar event details if requested")


async def db_write_worker_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    DB Write Sub-Agent:
    - Handles state mutations and HITL write flows (task creation, issue creation, calendar events, report schedulers).
    - Triggers langgraph.types.interrupt(...) for user confirmation before executing state writes.
    - Wrapped in try/except for graceful error reporting.
    """
    _emit_stream_status(status_text="DB Write worker preparing internal actions...")
    project_id = state.get("project_id") or ""
    user_id = state.get("user_id") or ""
    messages = state.get("messages", [])
    user_query = _get_latest_user_text(messages)
    lines = ["### DB Write Sub-Agent Findings:"]

    try:
        # Retrieve attached PRD document from cache if present
        file_id = state.get("file_id")
        doc_context = ""
        if file_id and user_id:
            try:
                doc_data = await get_parsed_document_from_cache(user_id=user_id, file_id=file_id)
                if doc_data and doc_data.get("parsed_markdown"):
                    filename = doc_data.get("file_name", "Uploaded PRD")
                    doc_md = doc_data.get("parsed_markdown", "")
                    doc_context = f"\n\nATTACHED PRD / SPECIFICATION DOCUMENT ({filename}):\n{doc_md[:60000]}\n"
                    print(f"[DB WRITE WORKER] Ingested attached PRD '{filename}' ({len(doc_md)} chars) for task/issue extraction.")
            except Exception as doc_err:
                print(f"[DB WRITE WORKER] Error reading doc cache: {doc_err}")

        # Check if user query involves task creation, issue creation, or calendar event scheduling
        write_keywords = [
            "create", "add", "new task", "new issue", "schedule", "meeting", "calendar",
            "event", "make task", "generate task", "task-", "bulk", "issue-", "confirm",
            "those", "these", "assign", "yes", "insert", "extract", "tasks", "issues",
        ]
        lower_query = user_query.lower()

        if any(k in lower_query for k in write_keywords) or bool(doc_context):
            try:
                openai_api_key = os.getenv("OPENAI_API_KEY", "")
                extractor_model = os.getenv("EXTRACTOR_MODEL", "gpt-4.1-nano")
                extractor_llm = ChatOpenAI(
                    model=extractor_model,
                    openai_api_key=openai_api_key,
                    temperature=0.0,
                ).with_structured_output(DBWriteIntentExtraction)

                # Collect recent conversation context (last 4 messages) to resolve anaphoras ('these tasks', 'those')
                context_lines = []
                recent_msgs = messages[-4:] if len(messages) > 4 else messages
                for m in recent_msgs:
                    role = "User" if (isinstance(m, HumanMessage) or (isinstance(m, dict) and m.get("type") in ["user", "human"])) else "Assistant"
                    content = m.content if hasattr(m, "content") else (m.get("content") if isinstance(m, dict) else str(m))
                    context_lines.append(f"{role}: {content}")
                convo_context = "\n".join(context_lines)

                extraction_prompt = [
                    SystemMessage(content=(
                        f"Today's Date: {datetime.now().strftime('%Y-%m-%d')}.\n"
                        "You are the DB Write Action Extractor. Your job is to extract structured tasks, issues, or calendar events to be created in the database.\n\n"
                        "CRITICAL EXTRACTION RULES:\n"
                        "1. TASK CREATION TRIGGER:\n"
                        "   - If the user asks to create tasks, add tasks, make tasks, or says 'create tasks of those please !', 'create these tasks', 'create those', 'yes create', 'yes confirm', or 'extract tasks':\n"
                        "     * ALWAYS set is_task_creation=True.\n"
                        "     * Extract all actionable tasks from the attached PRD (if provided) OR from the Assistant's previous messages/tables in conversation history into the `tasks` list.\n"
                        "     * Populate each task with a clean, concise title, contextual description, priority ('high', 'medium', 'low'), and a short 1-2 word domain tag in `tag`.\n"
                        "2. ISSUE CREATION TRIGGER:\n"
                        "   - If the user asks to create issues, log bugs, or says 'create issues of those', 'create issues for these blockers':\n"
                        "     * ALWAYS set is_issue_creation=True.\n"
                        "     * Extract all bugs/issues from the document or previous messages into `issues` with title, description, and severity ('critical', 'high', 'medium', 'low').\n"
                        "3. TITLES, DESCRIPTIONS & SHORT DYNAMIC TAGS:\n"
                        "   - Keep task titles concise and actionable (e.g. 'Fix dispute-hold release bug', 'Setup middleware routes'). Never return an empty task list if items were discussed or exist in the PRD.\n"
                        "   - For each task, ALWAYS generate a specific, short 1-2 word tag in `tag` based on its topic (e.g., 'Payment', 'Auth', 'Middleware', 'UI', 'Backend', 'DevOps', 'Client', 'Setup', 'Bugfix', 'API', 'Billing', 'Testing'). NEVER use 'PRD-Import' or generic placeholders."
                    )),
                    HumanMessage(content=f"Recent Conversation History:\n{convo_context}{doc_context}\n\nLatest User Query: '{user_query}'"),
                ]
                extracted: DBWriteIntentExtraction = await extractor_llm.ainvoke(extraction_prompt)

                # ── 1. Handle Task Creation HITL Interrupt ──────────────────
                if extracted.is_task_creation and extracted.tasks:
                    task_items_preview = []
                    for t in extracted.tasks:
                        clean_tag = (t.tag or "").replace("named-", "").strip()
                        if not clean_tag or clean_tag.lower() in ["prd-import", "prd_import", "prd import"]:
                            # Infer short tag from title
                            t_low = t.title.lower()
                            if any(k in t_low for k in ["pay", "bill", "money", "financ", "subscri"]):
                                clean_tag = "Payment"
                            elif any(k in t_low for k in ["auth", "login", "jwt", "user", "signup", "iam"]):
                                clean_tag = "Auth"
                            elif any(k in t_low for k in ["middle", "rout"]):
                                clean_tag = "Middleware"
                            elif any(k in t_low for k in ["api", "back", "database", "sql", "server"]):
                                clean_tag = "API"
                            elif any(k in t_low for k in ["client", "meet"]):
                                clean_tag = "Client"
                            elif any(k in t_low for k in ["ui", "front", "design", "css"]):
                                clean_tag = "UI"
                            elif any(k in t_low for k in ["bug", "fix", "error"]):
                                clean_tag = "Bugfix"
                            elif any(k in t_low for k in ["setup", "config", "install"]):
                                clean_tag = "Setup"
                            else:
                                clean_tag = "Feature"

                        task_items_preview.append({
                            "title": t.title.replace("named-", "").strip(),
                            "description": t.description if (t.description and not t.description.startswith("Task requested")) else "",
                            "priority": t.priority if t.priority in ["high", "medium", "low"] else "medium",
                            "tag": clean_tag,
                        })

                    tool_name = "bulk_create_tasks" if len(task_items_preview) > 1 else "create_task"
                    interrupt_payload = {
                        "tool": tool_name,
                        "message": f"Review and approve {len(task_items_preview)} task(s) to create:",
                        "preview": {
                            "type": "task",
                            "tasks": task_items_preview,
                        },
                    }

                    print("\n" + "=" * 70)
                    print(f"[HITL GRAPH_INTERRUPT] 🛑 PAUSING GRAPH FOR TASK CREATION APPROVAL")
                    print(f"[HITL TOOL NAME] {tool_name}")
                    print(f"[HITL PREVIEW CONTENT] Payload:\n{json.dumps(interrupt_payload, indent=2)}")
                    print("=" * 70 + "\n")

                    _emit_stream_status(status_text="Awaiting your approval to create tasks...", tool_called=tool_name, caller="DB Write agent")

                    # Pause graph execution and wait for user approval card interaction
                    resume_response = interrupt(interrupt_payload)

                    print("\n" + "=" * 70)
                    print(f"[HITL GRAPH_RESUME] ✅ TASK APPROVAL RECEIVED & GRAPH RESUMED!")
                    print(f"[HITL RESUME DATA RECEIVED]\n{json.dumps(resume_response, indent=2) if isinstance(resume_response, dict) else resume_response}")
                    print("=" * 70 + "\n")

                    if isinstance(resume_response, dict) and resume_response.get("action") == "cancel":
                        lines.append(f"- **Task Creation**: ❌ Cancelled by user.")
                    else:
                        edits = resume_response.get("edits") if isinstance(resume_response, dict) else None
                        final_tasks = edits if isinstance(edits, list) and edits else task_items_preview
                        tasks_for_convex = []
                        for t in final_tasks:
                            raw_tag = t.get("tag") or (t.get("type", {}).get("label") if isinstance(t.get("type"), dict) else None) or ""
                            tag_label = str(raw_tag).replace("named-", "").strip()
                            if not tag_label or tag_label.lower() in ["prd-import", "prd_import", "prd import"]:
                                t_low = str(t.get("title", "")).lower()
                                if any(k in t_low for k in ["pay", "bill", "money", "financ", "subscri"]):
                                    tag_label = "Payment"
                                elif any(k in t_low for k in ["auth", "login", "jwt", "user", "signup", "iam"]):
                                    tag_label = "Auth"
                                elif any(k in t_low for k in ["middle", "rout"]):
                                    tag_label = "Middleware"
                                elif any(k in t_low for k in ["api", "back", "database", "sql", "server"]):
                                    tag_label = "API"
                                elif any(k in t_low for k in ["client", "meet"]):
                                    tag_label = "Client"
                                elif any(k in t_low for k in ["ui", "front", "design", "css"]):
                                    tag_label = "UI"
                                elif any(k in t_low for k in ["bug", "fix", "error"]):
                                    tag_label = "Bugfix"
                                elif any(k in t_low for k in ["setup", "config", "install"]):
                                    tag_label = "Setup"
                                else:
                                    tag_label = "Feature"

                            tasks_for_convex.append({
                                "title": t.get("title", "Task").replace("named-", "").strip(),
                                "description": t.get("description", "") or "",
                                "priority": t.get("priority", "medium"),
                                "type": {
                                    "label": tag_label,
                                    "color": _get_tag_color(tag_label),
                                },
                            })
                        write_res = await write_bulk_tasks_to_convex({"projectId": project_id, "tasks": tasks_for_convex})
                        print(f"[HITL WRITE RESULT] Convex bulkInsertTasks response: {write_res}")
                        lines.append(f"- **Task Creation Status**: {write_res} ({len(tasks_for_convex)} task(s) created in project)")

                # ── 2. Handle Issue Creation HITL Interrupt ─────────────────
                elif extracted.is_issue_creation and extracted.issues:
                    issue_items_preview = [
                        {
                            "title": iss.title.replace("named-", "").strip(),
                            "description": iss.description if (iss.description and not iss.description.startswith("Issue logged")) else "",
                            "priority": iss.priority if iss.priority in ["high", "medium", "low"] else "medium",
                        }
                        for iss in extracted.issues
                    ]
                    tool_name = "bulk_create_issues" if len(issue_items_preview) > 1 else "create_issue"
                    interrupt_payload = {
                        "tool": tool_name,
                        "message": f"Review and approve {len(issue_items_preview)} issue(s) to create:",
                        "preview": {
                            "type": "issue",
                            "issues": issue_items_preview,
                        },
                    }

                    print("\n" + "=" * 70)
                    print(f"[HITL GRAPH_INTERRUPT] 🛑 PAUSING GRAPH FOR ISSUE CREATION APPROVAL")
                    print(f"[HITL TOOL NAME] {tool_name}")
                    print(f"[HITL PREVIEW CONTENT] Payload:\n{json.dumps(interrupt_payload, indent=2)}")
                    print("=" * 70 + "\n")

                    _emit_stream_status(status_text="Awaiting your approval to create issues...", tool_called=tool_name, caller="DB Write agent")

                    resume_response = interrupt(interrupt_payload)

                    print("\n" + "=" * 70)
                    print(f"[HITL GRAPH_RESUME] ✅ ISSUE APPROVAL RECEIVED & GRAPH RESUMED!")
                    print(f"[HITL RESUME DATA RECEIVED]\n{json.dumps(resume_response, indent=2) if isinstance(resume_response, dict) else resume_response}")
                    print("=" * 70 + "\n")

                    if isinstance(resume_response, dict) and resume_response.get("action") == "cancel":
                        lines.append(f"- **Issue Creation**: ❌ Cancelled by user.")
                    else:
                        edits = resume_response.get("edits") if isinstance(resume_response, dict) else None
                        final_issues = edits if isinstance(edits, list) and edits else issue_items_preview
                        issues_for_convex = []
                        for i in final_issues:
                            raw_p = (i.get("priority") or i.get("severity") or "medium").lower()
                            sev = "critical" if raw_p in ["critical", "high"] else "low" if raw_p == "low" else "medium"
                            issues_for_convex.append({
                                "title": i.get("title", "Issue").replace("named-", "").strip(),
                                "description": i.get("description", "") or "",
                                "severity": sev,
                                "environment": "dev",
                            })
                        write_res = await write_bulk_issues_to_convex({"projectId": project_id, "issues": issues_for_convex})
                        print(f"[HITL WRITE RESULT] Convex bulkInsertIssues response: {write_res}")
                        lines.append(f"- **Issue Creation Status**: {write_res} ({len(issues_for_convex)} issue(s) created in project)")

                # ── 3. Handle Calendar Event HITL Interrupt ─────────────────
                elif extracted.is_calendar_event and extracted.calendar_event:
                    cal = extracted.calendar_event
                    start_iso = cal.start_iso or datetime.now().strftime("%Y-%m-%dT10:00:00")
                    end_iso = cal.end_iso or datetime.now().strftime("%Y-%m-%dT11:00:00")
                    interrupt_payload = {
                        "tool": "create_calendar_event",
                        "message": "Review and confirm calendar event:",
                        "preview": {
                            "title": cal.title,
                            "description": cal.description or "Scheduled via Kaya PM",
                            "type": cal.event_type if cal.event_type in ["event", "milestone"] else "event",
                            "start": start_iso,
                            "end": end_iso,
                            "allDay": cal.all_day or False,
                        },
                    }

                    print("\n" + "=" * 70)
                    print(f"[HITL GRAPH_INTERRUPT] 🛑 PAUSING GRAPH FOR CALENDAR EVENT APPROVAL")
                    print(f"[HITL TOOL NAME] create_calendar_event")
                    print(f"[HITL PREVIEW CONTENT] Payload:\n{json.dumps(interrupt_payload, indent=2)}")
                    print("=" * 70 + "\n")

                    _emit_stream_status(status_text="Awaiting your approval to schedule calendar event...", tool_called="create_calendar_event", caller="DB Write agent")

                    resume_response = interrupt(interrupt_payload)

                    print("\n" + "=" * 70)
                    print(f"[HITL GRAPH_RESUME] ✅ CALENDAR APPROVAL RECEIVED & GRAPH RESUMED!")
                    print(f"[HITL RESUME DATA RECEIVED]\n{json.dumps(resume_response, indent=2) if isinstance(resume_response, dict) else resume_response}")
                    print("=" * 70 + "\n")

                    if isinstance(resume_response, dict) and resume_response.get("action") == "cancel":
                        lines.append(f"- **Calendar Event**: ❌ Cancelled by user.")
                    else:
                        preview_data = interrupt_payload["preview"]
                        if isinstance(resume_response, dict) and resume_response.get("edits"):
                            preview_data.update(resume_response["edits"])
                        cal_payload = {
                            "projectId": project_id,
                            "userId": user_id,
                            "title": preview_data["title"],
                            "description": preview_data["description"],
                            "eventType": preview_data["type"],
                            "startISO": preview_data["start"],
                            "endISO": preview_data["end"],
                            "allDay": preview_data["allDay"],
                        }
                        write_res = await write_calendar_event_to_convex(cal_payload)
                        print(f"[HITL WRITE RESULT] Convex createCalendarEvent response: {write_res}")
                        lines.append(f"- **Calendar Event Status**: {write_res}")

            except GraphInterrupt:
                raise
            except Exception as extract_err:
                print(f"[DB WRITE EXTRACT WARNING] {extract_err}")

        if len(lines) == 1:
            lines.append("- **DB Write Status**: Ready for write/mutation commands.")

        summary_text = "\n".join(lines)

        return {
            "_db_write_messages": [RESET_SENTINEL, {"role": "db_write", "content": summary_text}],
            "file_id": file_id,
        }

    except GraphInterrupt:
        raise
    except Exception as e:
        error_msg = f"DB Write Sub-Agent encountered an error: {e}"
        print(f"[DB WRITE WORKER ERROR] {error_msg}")
        return {
            "_db_write_messages": [
                RESET_SENTINEL,
                {"role": "db_write", "content": f"⚠️ [DB Write Notice]: {error_msg}"},
            ],
            "active_error": error_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-AGENT 3: MCP INTEGRATIONS WORKER (Slack, Calendly, Linear, Notion, Sentry)
# ─────────────────────────────────────────────────────────────────────────────

async def mcp_worker_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    MCP Sub-Agent:
    - Dynamically retrieves active connected SaaS apps & decrypted tokens from Convex.
    - Employs Smart Tool Pruning based on user query intent (Slack, Calendly, Linear, Notion, Sentry).
    - Executes MCP tools with timeouts and HITL awareness.
    - Normalized into structured Markdown findings in _mcp_messages.
    - Zero cascading failures via try/except isolation.
    """
    _emit_stream_status(status_text="MCP Agent querying connected workspace apps...")
    project_id = state.get("project_id") or ""
    user_name = state.get("user_name") or ""
    messages = state.get("messages", [])
    user_query = _get_latest_user_text(messages)

    if not project_id:
        return {
            "_mcp_messages": [
                RESET_SENTINEL,
                {"role": "mcp", "content": "⚠️ No active project selected. Cannot query MCP connectors."},
            ]
        }

    try:
        # Execute MCP agent workflow with smart pruning, live token fetching, and declarative skill execution
        mcp_result = await execute_mcp_agent_workflow(
            project_id=project_id,
            user_query=user_query,
            user_name=user_name,
            active_skill_content=state.get("active_skill_content"),
            selected_skill=state.get("selected_skill"),
        )

        summary = mcp_result.get("summary", "No MCP data retrieved.")
        executed_tools = mcp_result.get("executed_tools", [])

        for t_name in executed_tools:
            _emit_stream_status(tool_called=t_name, caller="MCP Agent")

        return {
            "_mcp_messages": [RESET_SENTINEL, {"role": "mcp", "content": summary}],
            "mcp_insights": mcp_result,
        }
    except GraphInterrupt:
        raise
    except Exception as e:
        error_msg = f"MCP Sub-Agent encountered an error: {e}"
        print(f"[MCP WORKER ERROR] {error_msg}")
        return {
            "_mcp_messages": [
                RESET_SENTINEL,
                {"role": "mcp", "content": f"⚠️ [MCP Integrations Notice]: {error_msg}"},
            ],
            "active_error": error_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-AGENT 4: GITHUB WORKER (PRs, Issues, Contributor Velocity, Release & CI Health)
# ─────────────────────────────────────────────────────────────────────────────


async def resolve_project_github_context(state: SupervisorState) -> Tuple[str, str, str, str]:
    """
    Resolves (owner, repo, token, username) for the active project.
    """
    token = state.get("github_access_token") or ""
    repo_full = state.get("repo_full_name") or ""
    username = state.get("github_username") or ""

    if not repo_full:
        repo_full = os.getenv("GITHUB_REPO", "").strip()

    owner, repo = "", ""
    if repo_full and "/" in repo_full:
        parts = repo_full.split("/", 1)
        owner, repo = parts[0].strip(), parts[1].strip()

    return owner, repo, token, username


async def github_worker_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    GitHub Sub-Agent (Powered by gpt-4.1-mini):
    - Strictly bound to the project's connected repository (owner/repo).
    - Determines and executes exactly the needed tools from:
        1. get_pull_requests_summary: PR review status, stale PRs (>48h), draft PRs, and blockers.
        2. get_issues: Open/closed issues, unassigned bugs, and blocker labels (excluding PRs).
        3. get_contributor_activity: Developer velocity, 7d vs 14d commit trends, and active contributors.
        4. get_release_and_ci_status: Release tags, milestones, and GitHub Actions CI workflow runs.
    - Runs tools in PARALLEL via asyncio.gather.
    - Formats findings into structured Markdown tables with item counts and clickable links.
    """
    _emit_stream_status(status_text="GitHub sub-agent is analyzing repository activity & PR health...")
    messages = state.get("messages", [])
    user_query = _get_latest_user_text(messages).lower()

    owner, repo, token, username = await resolve_project_github_context(state)

    if not owner or not repo:
        notice = (
            "ℹ️ **No GitHub Repository Linked**: No GitHub repository is connected to this project, "
            "or GitHub is not linked in your profile. Please link a repository in **Project Settings** "
            "to view Pull Requests, Issues, Contributor Velocity, and CI/CD Build Health."
        )
        return {
            "_github_messages": [RESET_SENTINEL, {"role": "github", "content": notice}],
            "github_insights": {"status": "not_connected", "message": notice},
        }

    # Determine tool requirements based on user query (strict whole-word matching)
    has_pr_kw = contains_exact_words(user_query, ["pr", "prs", "pull request", "pull requests", "code review", "stale pr", "stale prs", "merge", "branch", "branches"])
    has_issue_kw = contains_exact_words(user_query, ["github issue", "github issues", "github bug", "github bugs", "unassigned issue", "unassigned bug"]) or ("github" in user_query and any(k in user_query for k in ["issue", "issues", "bug", "bugs"]))
    has_contrib_kw = contains_exact_words(user_query, ["contributor", "contributors", "velocity", "commit", "commits", "author", "activity", "churn", "who is working", "code activity"])
    has_ci_kw = contains_exact_words(user_query, ["ci", "build", "workflow", "action", "actions", "run", "runs", "release", "releases", "tag", "tags", "pipeline"])
    is_general_query = not (has_pr_kw or has_issue_kw or has_contrib_kw or has_ci_kw)

    need_prs = has_pr_kw or is_general_query
    need_issues = has_issue_kw or is_general_query
    need_contrib = has_contrib_kw or is_general_query
    need_ci = has_ci_kw or is_general_query


    reasoning_parts = []
    if need_prs:
        reasoning_parts.append("pull requests & review bottlenecks")
    if need_issues:
        reasoning_parts.append("repository open & unassigned issues")
    if need_contrib:
        reasoning_parts.append("contributor commit velocity & recent code activity")
    if need_ci:
        reasoning_parts.append("release milestones & CI/CD workflow health")

    gh_reasoning = f"Querying repository '{owner}/{repo}' for {', '.join(reasoning_parts)}."
    _emit_stream_status(status_text=f"Inspecting '{owner}/{repo}'...", reasoning=gh_reasoning)

    if need_prs:
        _emit_stream_status(tool_called="get_pull_requests_summary", caller="GitHub sub-agent")
    if need_issues:
        _emit_stream_status(tool_called="get_issues", caller="GitHub sub-agent")
    if need_contrib:
        _emit_stream_status(tool_called="get_contributor_activity", caller="GitHub sub-agent")
    if need_ci:
        _emit_stream_status(tool_called="get_release_and_ci_status", caller="GitHub sub-agent")

    tasks = []
    task_keys = []

    if need_prs:
        tasks.append(fetch_github_pull_requests_async(owner, repo, token))
        task_keys.append("prs")
    if need_issues:
        tasks.append(fetch_github_issues_async(owner, repo, token))
        task_keys.append("issues")
    if need_contrib:
        tasks.append(fetch_github_contributor_activity_async(owner, repo, token, days=14))
        task_keys.append("contributors")
    if need_ci:
        tasks.append(fetch_github_release_and_ci_status_async(owner, repo, token))
        task_keys.append("release_ci")

    results = await asyncio.gather(*tasks, return_exceptions=True)
    raw_res = dict(zip(task_keys, results))

    lines = [f"## 🐙 GitHub Repository Findings: `{owner}/{repo}`\n"]

    # 1. Pull Requests Section
    prs_data = raw_res.get("prs")
    if isinstance(prs_data, dict) and not prs_data.get("error"):
        open_prs = prs_data.get("open_prs", [])
        stale_prs = prs_data.get("stale_prs", [])
        merged_prs = prs_data.get("recent_merged_prs", [])

        lines.append(f"### 🔀 Pull Requests (Total Open: {len(open_prs)} | Stale >48h: {len(stale_prs)})")
        if open_prs:
            lines.append("| # | PR Title | Author | State | Draft | Stale (>48h) | Branches | Reviewers | Link |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for p in open_prs:
                stale_badge = "⚠️ **Yes (>48h)**" if p.get("is_stale") else "No"
                draft_badge = "Draft" if p.get("draft") else "Ready"
                branches = f"`{p.get('head_branch', '')}` → `{p.get('base_branch', '')}`"
                revs = ", ".join(f"@{r}" for r in p.get("requested_reviewers", [])) or "None"
                link = f"[PR #{p.get('number')}]({p.get('html_url')})" if p.get("html_url") else f"#{p.get('number')}"
                lines.append(f"| **#{p.get('number')}** | {p.get('title')} | @{p.get('author')} | `{p.get('state')}` | {draft_badge} | {stale_badge} | {branches} | {revs} | {link} |")
        else:
            lines.append("No active open pull requests found.")

        if merged_prs:
            lines.append("\n**Recently Merged PRs:** " + ", ".join(f"[#{m['number']} - {m['title']}]({m['html_url']})" for m in merged_prs))
        lines.append("")
    elif isinstance(prs_data, dict) and prs_data.get("error"):
        lines.append(f"⚠️ *Pull Requests fetch notice: {prs_data.get('error')}*\n")

    # 2. Issues Section
    issues_data = raw_res.get("issues")
    if isinstance(issues_data, dict) and not issues_data.get("error"):
        open_issues = issues_data.get("open_issues", [])
        unassigned = issues_data.get("unassigned_issues", [])
        blockers = issues_data.get("blocker_issues", [])

        lines.append(f"### 🐛 GitHub Issues (Total Open: {len(open_issues)} | Unassigned: {len(unassigned)} | Blockers: {len(blockers)})")
        if open_issues:
            lines.append("| # | Issue Title | Author | State | Assignees | Priority / Blocker | Comments | Link |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for iss in open_issues[:15]:
                assignees = ", ".join(f"@{a}" for a in iss.get("assignees", [])) or "*(Unassigned)*"
                blocker_badge = "🚨 **Blocker/Bug**" if iss.get("is_blocker") else "Normal"
                link = f"[Issue #{iss.get('number')}]({iss.get('html_url')})" if iss.get("html_url") else f"#{iss.get('number')}"
                lines.append(f"| **#{iss.get('number')}** | {iss.get('title')} | @{iss.get('author')} | `{iss.get('state')}` | {assignees} | {blocker_badge} | {iss.get('comments_count', 0)} | {link} |")
        else:
            lines.append("No open GitHub issues found.")
        lines.append("")
    elif isinstance(issues_data, dict) and issues_data.get("error"):
        lines.append(f"⚠️ *Issues fetch notice: {issues_data.get('error')}*\n")

    # 3. Contributor Activity Section
    contrib_data = raw_res.get("contributors")
    if isinstance(contrib_data, dict) and not contrib_data.get("error"):
        top_contribs = contrib_data.get("top_contributors", [])
        recent_commits = contrib_data.get("recent_commits", [])
        commits_7d = contrib_data.get("commits_last_7_days", 0)
        total_commits = contrib_data.get("total_commits", 0)

        lines.append(f"### 📈 Developer Velocity & Contributor Activity (Last 14 Days: {total_commits} commits | Last 7 Days: {commits_7d} commits)")
        if top_contribs:
            lines.append("| Contributor | Commits (14d) | Recent Activity |")
            lines.append("| :--- | :--- | :--- |")
            for c in top_contribs[:6]:
                last_dt = c.get("last_commit_date", "")[:10] if c.get("last_commit_date") else "Recent"
                lines.append(f"| **@{c.get('login')}** | {c.get('commit_count')} commits | Last committed {last_dt} |")
        if recent_commits:
            lines.append("\n**Recent Commits:**")
            for rc in recent_commits[:5]:
                c_link = f"[`{rc.get('sha')}`]({rc.get('html_url')})" if rc.get("html_url") else f"`{rc.get('sha')}`"
                lines.append(f"- {c_link} {rc.get('message')} (@{rc.get('author')})")
        lines.append("")
    elif isinstance(contrib_data, dict) and contrib_data.get("error"):
        lines.append(f"⚠️ *Contributor activity notice: {contrib_data.get('error')}*\n")

    # 4. Release & CI Status Section
    ci_data = raw_res.get("release_ci")
    if isinstance(ci_data, dict) and not ci_data.get("error"):
        runs = ci_data.get("recent_workflow_runs", [])
        ci_health = ci_data.get("ci_health", {})
        latest_rel = ci_data.get("latest_release")

        health_badge = "✅ All Passing" if ci_health.get("is_healthy") else f"⚠️ {ci_health.get('failing_runs', 0)} Failing"
        lines.append(f"### 🚀 Release & CI/CD Status (Build Health: {health_badge})")
        if latest_rel:
            lines.append(f"**Latest Release:** [{latest_rel.get('tagName')} - {latest_rel.get('name')}]({latest_rel.get('htmlUrl')}) (Published: {str(latest_rel.get('publishedAt'))[:10]})")
        if runs:
            lines.append("| Workflow Name | Branch | Status | Conclusion | Created | Link |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for r in runs[:6]:
                conc = r.get("conclusion") or r.get("status")
                conc_icon = "✅ success" if conc == "success" else "❌ failure" if conc in ["failure", "timed_out"] else f"⏳ {conc}"
                r_link = f"[Run #{r.get('id')}]({r.get('htmlUrl')})" if r.get("htmlUrl") else f"#{r.get('id')}"
                lines.append(f"| **{r.get('name')}** | `{r.get('headBranch')}` | `{r.get('status')}` | {conc_icon} | {str(r.get('createdAt'))[:10]} | {r_link} |")
        lines.append("")
    elif isinstance(ci_data, dict) and ci_data.get("error"):
        lines.append(f"⚠️ *Release & CI fetch notice: {ci_data.get('error')}*\n")

    summary_text = "\n".join(lines)

    return {
        "_github_messages": [RESET_SENTINEL, {"role": "github", "content": summary_text}],
        "github_insights": {
            "owner": owner,
            "repo": repo,
            "pull_requests": prs_data if isinstance(prs_data, dict) else {},
            "issues": issues_data if isinstance(issues_data, dict) else {},
            "contributors": contrib_data if isinstance(contrib_data, dict) else {},
            "release_ci": ci_data if isinstance(ci_data, dict) else {},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. KAYA SYNTHESIZER & DIRECT NODES (Only Kaya streams tokens to user)
# ─────────────────────────────────────────────────────────────────────────────

KAYA_SYSTEM_BASE = """You are Kaya, an Executive AI Technical Product Manager for the WEKRAFT Platform.
Your Persona & Standards:
1. Speak professionally, directly, and supportively like a Senior Technical Product Manager.
2. Be concise and crisp — avoid unnecessary introductory/closing fluff or verbose filler.
3. Be highly data-driven: always reference specific project task names, issue titles, priorities, assignees, and real metrics provided in the context.
4. Structure information clearly using concise Markdown bullet points, bold key terms, and mini tables when reporting status.
5. If medical, legal, or raw coding queries arise, give a brief, polite, professional deflection that you focus purely on technical project execution.
"""


def get_direct_chat_llm() -> ChatOpenAI:
    """Returns Groq 120b LLM for fast direct conversation/chit-chat when no sub-agents are invoked."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    max_tokens = int(os.getenv("KAYA_MAX_TOKENS", "700"))
    if groq_key:
        groq_model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
        return ChatOpenAI(
            model=groq_model,
            openai_api_key=groq_key,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.2,
            max_tokens=max_tokens,
            streaming=True,
        )
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    return ChatOpenAI(
        model="gpt-4.1-mini",
        openai_api_key=openai_key or "none",
        temperature=0.2,
        max_tokens=max_tokens,
        streaming=True,
    )


def get_synthesizer_llm() -> ChatOpenAI:
    """Returns gpt-4.1-mini LLM to synthesize sub-agent findings into high-impact PM responses."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    max_tokens = int(os.getenv("KAYA_MAX_TOKENS", "1500"))
    if openai_key:
        return ChatOpenAI(
            model="gpt-4.1-mini",
            openai_api_key=openai_key,
            temperature=0.2,
            max_tokens=max_tokens,
            streaming=True,
        )
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    return ChatOpenAI(
        model=os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b"),
        openai_api_key=groq_key or "none",
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=max_tokens,
        streaming=True,
    )


async def kaya_direct_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Kaya Direct Response Node: Fast conversational greeting/chit-chat using Groq 120b with token streaming."""
    _emit_stream_status(status_text="Kaya is typing...")
    user_name = state.get("user_name") or "there"
    project_name = state.get("project_name") or "your active project"
    project_deadline = state.get("project_deadline") or "No deadline currently set"
    current_date = datetime.now().strftime("%A, %B %d, %Y")

    system_prompt = f"""You are Kaya, an Executive Technical Project Manager on the WEKRAFT Platform.
You are actively managing the project "{project_name}".

CONVERSATION CONTEXT:
- Today's Real-World Date: {current_date}
- Project Name: {project_name}
- Project Target Deadline: {project_deadline}
- User Name: {user_name}

YOUR OPERATING PRINCIPLES:
1. Greet {user_name} professionally and warmly, acknowledging your role as the Technical PM for "{project_name}".
2. Keep your direct response concise (2-3 sentences), executive, and focused on helping them drive the project forward today ({current_date}).
3. ZERO Raw Database IDs: Under NO circumstances should you output, mention, or leak alphanumeric database IDs (such as 'kn71qtgem...' or 'nd7088...'). Always refer strictly to the project name and user name.
4. If medical, legal, or non-project topics are asked, provide a polite, one-sentence deflection that you focus purely on technical project execution.
"""

    llm = get_direct_chat_llm()

    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    response = await llm.ainvoke(messages)

    return {"messages": [response]}


async def kaya_synthesizer_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Kaya Synthesizer Node:
    - Ingests all sub-agent worker outputs (_analyst_messages, _db_write_messages, _mcp_messages).
    - Injects project details, deadline awareness directly from state, and current date.
    - Synthesizes findings using gpt-4.1-mini, streaming executive PM tokens to user.
    """
    _emit_stream_status(status_text="Kaya is synthesizing executive PM insights...")
    user_name = state.get("user_name") or "there"
    current_date = datetime.now().strftime("%A, %B %d, %Y")

    # Project deadline and name awareness check directly from state
    project_name = state.get("project_name") or "Active Project"
    raw_deadline = state.get("project_deadline")

    if raw_deadline and raw_deadline != "No deadline currently set":
        deadline_text = str(raw_deadline)
        try:
            from dateutil import parser as dt_parser
            d_dt = dt_parser.parse(str(raw_deadline))
            now_dt = datetime.now()
            days_diff = (d_dt.date() - now_dt.date()).days
            if days_diff >= 0:
                deadline_text = f"{raw_deadline} ({days_diff} day(s) remaining from today)"
            else:
                deadline_text = f"{raw_deadline} ({abs(days_diff)} day(s) overdue)"
        except Exception:
            pass
    else:
        deadline_text = "No deadline currently set"

    # Collect findings from all sub-agents (Internal DB + External MCP + GitHub)
    findings_blocks = []
    for key, label in [
        ("_analyst_messages", "ANALYST FINDINGS"),
        ("_db_write_messages", "DB WRITE ACTIONS"),
        ("_github_messages", "GITHUB REPOSITORY FINDINGS (PRs, Code Reviews, Contributor Velocity, Release & CI Health)"),
        ("_mcp_messages", "THIRD-PARTY MCP INTEGRATION FINDINGS (Sentry, Jira, Vercel, Slack, Linear, Notion, ETC)"),
    ]:
        msgs = state.get(key, [])
        if msgs:
            content = msgs[-1].get("content") if isinstance(msgs[-1], dict) else str(msgs[-1])
            findings_blocks.append(f"--- {label} ---\n{content}")

    findings_prompt = "\n\n".join(findings_blocks) if findings_blocks else "No worker findings generated."

    active_error = state.get("active_error")
    error_instructions = f"\nNote: Active service error notice: {active_error}. Acknowledge this gracefully." if active_error else ""

    active_skill_content = state.get("active_skill_content")
    selected_skill = state.get("selected_skill")
    skill_guidance = ""
    if active_skill_content and selected_skill:
        skill_guidance = f"\nACTIVE PROCEDURAL SKILL GUIDELINES ({selected_skill}):\n{active_skill_content.strip()}\n"

    system_prompt = f"""You are Kaya, an Executive Technical Project Manager on the WEKRAFT Platform.
You are actively managing the project "{project_name}".

CONVERSATION & TEMPORAL CONTEXT:
- Today's Real-World Date: {current_date}
- Project Name: {project_name}
- Project Target Deadline: {deadline_text}
- User Name: {user_name}
{error_instructions}
{skill_guidance}
YOUR PERSONA & STANDARDS:
1. Executive PM Tone: Speak with crisp, authoritative, supportive professionalism. Avoid generic fluff, filler phrases, or introductory throat-clearing.
2. Temporal Grounding & Proximity: Today is {current_date}. The target deadline is {deadline_text}. Use these real-world temporal anchors to evaluate timeline urgency, overdue risks, and sprint momentum without hallucinating dates.
3. Strict Privacy & Zero Raw Database IDs: NEVER output, mention, or leak internal Convex database IDs (such as 'kn71qtgem...' or 'nd7088...'). Use only the real project name ('{project_name}'), user name ('{user_name}'), and actual task/issue/ticket titles.
4. MANDATORY STRUCTURED MARKDOWN TABLES FOR INTEGRATIONS, ISSUES, TASKS & SPRINTS:
   - CLEAR ITEM COUNTS: Always start each integration or data section with a prominent header indicating the total item count (e.g. `### 🚨 Sentry Issues (Total: 20 Unresolved)` or `### 🚀 Vercel Deployments (Total: 3)` or `### 📋 Internal Tasks (Total: 3)` or `### 🐛 Internal Issues (Total: 2)`).
   - COMPREHENSIVE MARKDOWN TABLES: You MUST render ALL retrieved items in clean, formatted Markdown Tables with all relevant details:
     * Sentry: `| ID / Short Key | Issue / Error Title | Culprit / Location | Status | Level / Severity | Events | Users | Link |`
     * Jira / Linear: `| Key | Issue / Task Summary | Status | Priority | Assignee | Link |`
     * Vercel: `| Project | Deployment URL | State / Status | Commit / Branch | Target / Environment | Creator | Link |`
     * Internal Project Tasks: `| Task Title | Assignee | Priority | Deadline / Status |`
     * Internal Project Issues: `| Issue Title | Severity | Status | Assignees |`
     * Internal Sprints: `| Sprint Name | Goal | Status | Progress | Dates |`
   - CLICKABLE LINKS: If URLs, permalinks, inspector URLs, or deployment links are available in the worker data, you MUST include clickable Markdown links (e.g. `[View on Sentry](https://...)` or `[Open Deployment](https://...)`) in the Link column.
   - DO NOT COLLAPSE OR TRUNCATE: List every single issue/item in its own table row. Never use placeholder lines like '| ... and 15 more |'.
   - EXECUTIVE SUMMARY AT END: At the very end of your response, provide an authoritative '### 📊 Executive PM Summary & Next Actions' section containing:
     * **Health & Volume Overview**: A crisp breakdown of totals, system stability, and deployment state.
     * **Top 3 Most Critical Items / Blockers**: The top 3 issues that need immediate developer attention (highlighting error frequency, critical priority, or failed deployments).
     * **Recommended Action Plan**: 2-3 concrete next steps for the engineering team.
5. STRICT ZERO-HALLUCINATION & GROUND-TRUTH RULE: Under NO circumstances should you fabricate, simulate, or invent tasks, issues, ticket keys, epics, bug titles, or member assignments that are not present in the SUB-AGENT WORKER DATA. If an integration returns 0 items or returns an error/unauthorized status, explicitly inform the user of that exact status and advise them to reconnect in the Integrations tab. Never invent placeholder tickets.
6. NEVER CLAIM WRITE ACTIONS WITHOUT CONFIRMATION: Never claim or state that tasks, issues, or calendar events 'have been created' or 'have been inserted' unless the SUB-AGENT WORKER DATA explicitly shows a successful write result (e.g. '✅ Bulk created ...'). If no database insertion occurred, do not claim tasks were created.
7. NOTION / EXTERNAL DOC CREATION CONFIRMATION & DIRECT LINK:
   If the SUB-AGENT WORKER DATA indicates that a Notion page was created or includes a Notion URL:
   - You MUST prominently display the confirmation and the direct clickable link at the VERY TOP of your response:
     `### 📄 Created Notion Doc Page: [Open Notion Document](<url>)`
   - Summarize the contents that were placed into the Notion document in clean, readable Markdown (including a breakdown of tasks and notes included).
   - NEVER output raw JSON request payloads or parameters in your chat response. Always format as professional Markdown.

SUB-AGENT WORKER DATA:
{findings_prompt}
"""

    llm = get_synthesizer_llm()

    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    response = await llm.ainvoke(messages)

    return {"messages": [response]}



# ─────────────────────────────────────────────────────────────────────────────
# 5. GRAPH COMPILATION
# ─────────────────────────────────────────────────────────────────────────────

def build_kaya_graph():
    """Builds and compiles the master Kaya multi-agent LangGraph workflow."""
    workflow = StateGraph(SupervisorState)

    # Add Nodes (4 Sub-Agents + Direct + Synthesizer)
    workflow.add_node("supervisor_router_node", supervisor_router_node)
    workflow.add_node("kaya_direct_node", kaya_direct_node)
    workflow.add_node("analyst_node", analyst_worker_node)
    workflow.add_node("db_write_node", db_write_worker_node)
    workflow.add_node("github_node", github_worker_node)
    workflow.add_node("mcp_node", mcp_worker_node)
    workflow.add_node("kaya_synthesizer_node", kaya_synthesizer_node)

    # Edge: Start -> Router
    workflow.add_edge(START, "supervisor_router_node")

    # Conditional Fan-Out Edge: Router -> Direct or Parallel Sub-Agents
    workflow.add_conditional_edges(
        "supervisor_router_node",
        route_supervisor,
        ["kaya_direct_node", "analyst_node", "db_write_node", "github_node", "mcp_node"],
    )

    # Direct Response terminates directly
    workflow.add_edge("kaya_direct_node", END)

    # Fan-In: All Sub-Agent Workers converge into Kaya Synthesizer
    workflow.add_edge("analyst_node", "kaya_synthesizer_node")
    workflow.add_edge("db_write_node", "kaya_synthesizer_node")
    workflow.add_edge("github_node", "kaya_synthesizer_node")
    workflow.add_edge("mcp_node", "kaya_synthesizer_node")

    # Synthesizer terminates at END
    workflow.add_edge("kaya_synthesizer_node", END)

    # Checkpointer for durable execution & HITL resumes (Redis checkpointer)
    checkpointer = get_checkpointer()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    print("[KAYA GRAPH] Master Supervisor & Multi-Agent Graph compiled successfully.")
    return compiled_graph


# Expose singleton compiled graph
kaya_graph = build_kaya_graph()

