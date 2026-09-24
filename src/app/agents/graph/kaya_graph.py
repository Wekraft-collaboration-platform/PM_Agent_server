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
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
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
    fetch_project_insights_async,
    fetch_scheduler_async,
    write_calendar_event_to_convex,
    write_sprint_to_convex,
    write_items_to_sprint,
    write_scheduler_to_convex,
    write_bulk_tasks_to_convex,
    write_bulk_issues_to_convex,
)
from app.agents.tools.mcp_client import execute_mcp_agent_workflow
from app.core.utils.document_parser import get_parsed_document_from_cache

load_dotenv(override=True)



# ─────────────────────────────────────────────────────────────────────────────
# 1. SUPERVISOR ROUTER DECISION SCHEMA & PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    actions: List[str] = Field(
        description=(
            "List of sub-agent actions to trigger in parallel (select ONLY the sub-agents explicitly required):\n"
            "- 'mcp': ONLY for third-party external integrations (explicitly requested Jira, Linear, Slack, Calendly, Notion, MCP). NEVER call for internal project tasks/issues/sprints or if user hasn't asked for third party.\n"
            "- 'db_write': DB Write agent for creating tasks, issues, calendar events, or report schedulers in this project\n"
            "- 'analyst': Project Analyst agent for internal project read analytics: user daily standup, task summaries, issue tracking, member workloads, project health, deadlines\n"
            "- 'sprint': Sprint agent for sprint insights/velocity, active sprint status, sprint creation, or backlog item assignments\n"
            "- 'direct_response': simple greeting, casual conversation, or general product manager chat"
        )
    )
    reasoning: str = Field(description="Clear reasoning for the routing decision.")


ROUTER_SYSTEM_PROMPT = """You are the primary Supervisor Router for the WEKRAFT AI Platform.
Analyze the user's incoming query (and previous query context if provided) and decide strictly which specialized sub-agent workers to trigger.

CRITICAL RULES:
1. CREATING TASKS, ISSUES, SPRINTS IN THIS PROJECT:
   - When the user asks to create, add, or generate tasks, issues, or calendar events (e.g., 'create 3 issues for these errors', 'create tasks', 'schedule meeting'), it ALWAYS targets this project's internal database -> route to 'db_write' (DB Write Agent).
   - When the user asks to create or plan sprints -> route to 'sprint' (Sprint Agent).
   - NEVER call 'mcp' for creating tasks, issues, or sprints unless the user explicitly specifies an external third-party destination (e.g., 'create issue in Jira', 'create ticket in Linear'). Even if issues are based on previous Sentry errors, creating them is an internal project action.

2. THIRD-PARTY / MCP AGENT ('mcp'):
   - ONLY call 'mcp' if the user EXPLICITLY asks for a third-party app or external tool (Jira, Linear, Slack, Calendly, Notion, external service, MCP), OR if the current query is a follow-up referring back to a third-party app mentioned in the previous query (e.g., Previous: 'get me all tasks from Jira', Current: 'i want issues also from that' -> 'mcp').
   - NEVER call 'mcp' for general project tasks, issues, sprints, or standups if no third-party integration was mentioned in either the current or previous query.

3. SPRINT AGENT ('sprint'):
   - ALWAYS call 'sprint' whenever the user asks about sprints (sprint progress, active sprint, sprint velocity, sprint tasks, sprint items, sprint creation, sprint planning), or follows up on previous sprint context.

4. MINIMAL SUB-AGENT CALLS:
   - NEVER call extra sub-agents if not asked by the user. Route ONLY to the exact sub-agents necessary to fulfill the user's request.
   - Example 1: 'create issues for these sentry errors' -> ['db_write'] (NO 'mcp').
   - Example 2: 'what are my tasks/issues and sprints' -> ['analyst', 'sprint'] (NO 'mcp').
   - Example 3: 'show my tasks and standup' -> ['analyst'] (NO 'sprint', NO 'mcp').
   - Example 4: 'how is the current sprint doing' -> ['sprint'] (NO 'analyst', NO 'mcp').
   - Example 5: 'check jira tickets and slack' -> ['mcp'].
   - Example 6: 'hi how are you' -> ['direct_response'].

Available Actions:
1. 'mcp': Third-party SaaS integrations ONLY (Jira, Linear, Slack, Calendly, Notion, Sentry, HubSpot, Vercel).
2. 'db_write': DB Write agent for project mutations: internal task creation, issue creation, calendar event scheduling, report scheduler setup.
3. 'analyst': Project Analyst agent for internal read-only analytics: User daily standup, task summaries, issue tracking, member workloads, project health, or project deadlines/timelines.
4. 'sprint': Sprint velocity, active sprint progress, sprint creation, or backlog item allocation.
5. 'direct_response': Greetings (hi, hello), casual conversation, or general questions requiring no live database queries.
"""


async def route_user_request(
    user_input: str,
    project_id: Optional[str] = None,
    has_attached_doc: bool = False,
) -> SupervisorDecision:
    """Evaluates user input using Groq 120b LLM router and outputs structured SupervisorDecision with deterministic safeguards."""
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    # Ensure 120b model is used for the router (openai/gpt-oss-120b)
    router_model = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b")

    doc_hint = "\n[Notice: An uploaded PRD/specification document is attached in this session.]" if has_attached_doc else ""

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
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

        decision: SupervisorDecision = await llm.ainvoke(messages)
        # Ensure fallback if actions is empty
        if not decision.actions:
            decision.actions = ["direct_response"]

        latest_text = user_input
        if "Current Query:" in user_input:
            latest_text = user_input.split("Current Query:")[-1].strip()
        lower_latest = latest_text.lower()
        lower_query = user_input.lower()

        # Document-specific routing rules
        if has_attached_doc or "[attached:" in lower_query or "prd" in lower_latest or "doc" in lower_latest:
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

        # Check for internal creation intent vs external third-party destination
        creation_verbs = ["create", "add", "make", "insert", "generate", "schedule", "new", "extract"]
        creation_nouns = ["task", "tasks", "issue", "issues", "bug", "bugs", "sprint", "sprints", "event", "meeting", "these", "those"]
        is_internal_creation = any(v in lower_latest for v in creation_verbs) and any(n in lower_latest for n in creation_nouns)
        mcp_explicit_targets = ["jira", "linear", "slack", "calendly", "notion", "hubspot", "vercel", "mcp", "atlassian"]
        has_explicit_mcp_in_latest = any(k in lower_latest for k in mcp_explicit_targets)

        # Rule 1: Internal creation routing (Task / Issue -> db_write, Sprint -> sprint)
        if is_internal_creation and not has_explicit_mcp_in_latest:
            if any(n in lower_latest for n in ["task", "tasks", "issue", "issues", "bug", "bugs", "event", "meeting", "these", "those"]):
                if "db_write" not in decision.actions:
                    decision.actions.append("db_write")
            if any(n in lower_latest for n in ["sprint", "sprints"]):
                if "sprint" not in decision.actions:
                    decision.actions.append("sprint")
            if "mcp" in decision.actions:
                decision.actions.remove("mcp")
            if "direct_response" in decision.actions:
                decision.actions.remove("direct_response")

        # Rule 2 Deterministic Safeguard: Never call MCP unless explicitly requested
        elif not has_explicit_mcp_in_latest:
            if "mcp" in decision.actions:
                decision.actions.remove("mcp")
                if not decision.actions:
                    decision.actions = ["analyst"]
        elif has_explicit_mcp_in_latest and "mcp" not in decision.actions:
            if "direct_response" in decision.actions:
                decision.actions.remove("direct_response")
            decision.actions.append("mcp")
            decision.reasoning += " (MCP auto-included due to explicit keyword match)"

        # Rule 3 Deterministic Safeguard: Always call sprint agent for sprint queries
        sprint_keywords = ["sprint", "sprints", "velocity", "backlog", "burndown"]
        if any(k in lower_latest for k in sprint_keywords):
            if "sprint" not in decision.actions:
                if "direct_response" in decision.actions:
                    decision.actions.remove("direct_response")
                decision.actions.append("sprint")
                decision.reasoning += " (Sprint agent auto-included due to sprint keyword match)"

        # Rule 4: Clean up direct_response if actionable subagents exist
        if len(decision.actions) > 1 and "direct_response" in decision.actions:
            decision.actions.remove("direct_response")

        safe_reasoning = decision.reasoning.encode("ascii", "replace").decode("ascii")
        print(f"[ROUTER DECISION] Actions: {decision.actions} | Reasoning: {safe_reasoning}")
        return decision

    except Exception as e:
        safe_err = str(e).encode("ascii", "replace").decode("ascii")
        print(f"[ROUTER WARNING] Fallback routing due to: {safe_err}")
        actions = []
        latest_text = user_input
        if "Current Query:" in user_input:
            latest_text = user_input.split("Current Query:")[-1].strip()
        lower_latest = latest_text.lower()

        if any(k in lower_latest for k in ["sprint", "sprints", "velocity"]):
            actions.append("sprint")
        if any(k in lower_latest for k in ["create", "add", "make", "insert", "schedule", "extract"]) and any(k in lower_latest for k in ["task", "tasks", "issue", "issues", "event", "these", "those"]):
            actions.append("db_write")
        elif any(k in lower_latest for k in ["task", "tasks", "issue", "issues", "standup", "workload", "health", "project", "critical", "prd", "doc"]):
            actions.append("analyst")
        if any(k in lower_latest for k in ["jira", "linear", "slack", "calendly", "notion", "hubspot", "vercel", "mcp"]):
            actions.append("mcp")
        if not actions:
            actions = ["direct_response"]

        return SupervisorDecision(
            actions=actions,
            reasoning=f"Fallback routing: {safe_err}",
        )


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
        if payload:
            writer(payload)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────

async def supervisor_router_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Evaluates user query and determines parallel execution branches."""
    messages = state.get("messages", [])
    user_query = _get_router_user_query(messages)
    project_id = state.get("project_id")
    file_id = state.get("file_id")

    _emit_stream_status(status_text="Kaya is thinking...")
    decision = await route_user_request(user_query, project_id, has_attached_doc=bool(file_id))

    # Immediately emit reasoning event so frontend displays it right after thinking
    _emit_stream_status(status_text="Kaya is reasoning...", reasoning=decision.reasoning)

    # Emit subagent delegation events for actions selected
    for action in decision.actions:
        if action == "analyst":
            _emit_stream_status(subagent_called="analyst_agent")
        elif action == "db_write":
            _emit_stream_status(subagent_called="db_write_agent")
        elif action == "sprint":
            _emit_stream_status(subagent_called="sprint_agent")
        elif action == "mcp":
            _emit_stream_status(subagent_called="mcp_agent")

    return {
        "next": decision.actions,
        "router_reasoning": decision.reasoning,
        "action_type": decision.actions[0] if len(decision.actions) == 1 else "multi_agent",
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
    if "sprint" in actions:
        target_nodes.append("sprint_node")
    if "mcp" in actions:
        target_nodes.append("mcp_node")

    return target_nodes if target_nodes else ["kaya_direct_node"]


# ─────────────────────────────────────────────────────────────────────────────
# SUB-AGENT 1: ANALYST WORKER (Parallel Read Tools + Zero Failure Crash)
# ─────────────────────────────────────────────────────────────────────────────

async def analyst_worker_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Analyst Sub-Agent:
    - Runs tools in PARALLEL via asyncio.gather (standup, tasks, issues, workloads, project timeline).
    - Normalizes results into non-empty structured Markdown findings.
    - Wrapped in try/except for graceful error reporting.
    """
    _emit_stream_status(status_text="Analyst worker analyzing project health & tasks in parallel...")
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
        # Determine parallel tool fetch requirements based on user query
        need_standup = any(k in user_query for k in ["standup", "my task", "assigned", "my issue", "today", "to do"])
        need_workload = any(k in user_query for k in ["workload", "team", "members", "who is working", "capacity"])

        # Emit live tool call events
        _emit_stream_status(tool_called="get_tasks_summary", caller="Project analyst")
        _emit_stream_status(tool_called="get_issues_summary", caller="Project analyst")
        _emit_stream_status(tool_called="get_project_insights", caller="Project analyst")
        if need_standup:
            _emit_stream_status(tool_called="get_user_standup", caller="Project analyst")
        if need_workload:
            _emit_stream_status(tool_called="get_member_workload", caller="Project analyst")

        # Execute read tools in parallel concurrently
        tasks_future = fetch_tasks_summary_async(project_id)
        issues_future = fetch_issues_summary_async(project_id)
        project_future = fetch_project_insights_async(project_id)
        standup_future = fetch_user_standup_async(project_id, user_id) if need_standup else asyncio.sleep(0, result={})
        workload_future = fetch_member_workload_async(project_id) if need_workload else asyncio.sleep(0, result={})

        tasks_res, issues_res, project_res, standup_res, workload_res = await asyncio.gather(
            tasks_future, issues_future, project_future, standup_future, workload_future, return_exceptions=True
        )

        # Normalize exceptions if any individual tool failed
        tasks_data = tasks_res if not isinstance(tasks_res, Exception) else {"error": str(tasks_res)}
        issues_data = issues_res if not isinstance(issues_res, Exception) else {"error": str(issues_res)}
        project_data = project_res if not isinstance(project_res, Exception) else {"error": str(project_res)}
        standup_data = standup_res if not isinstance(standup_res, Exception) else {"error": str(standup_res)}
        workload_data = workload_res if not isinstance(workload_res, Exception) else {"error": str(workload_res)}

        # Build clean structured summary for Kaya Synthesis
        lines = ["### Analyst Sub-Agent Findings:"]

        # Tasks Section
        if "error" in tasks_data:
            lines.append(f"- **Tasks Summary**: ⚠️ {tasks_data['error']}")
        else:
            total = tasks_data.get("totalCount", tasks_data.get("total", 0))
            completed = tasks_data.get("completedCount", 0)
            blocked = tasks_data.get("blockedCount", 0)
            active_tasks = tasks_data.get("criticalAndActiveTasks", [])
            in_prog = sum(1 for t in active_tasks if "prog" in str(t.get("status", "")).lower())
            todo = sum(1 for t in active_tasks if "start" in str(t.get("status", "")).lower() or "todo" in str(t.get("status", "")).lower())
            lines.append(f"- **Tasks Breakdown**: Total Tasks: {total} | In Progress: {in_prog} | To-Do: {todo} | Completed: {completed} | Blocked: {blocked}")
            if active_tasks:
                lines.append("  **Active Tasks List**:")
                for t in active_tasks:
                    title = t.get("title", "Untitled")
                    status = t.get("status", "unknown")
                    prio = t.get("priority", "normal")
                    assignees = ", ".join(t.get("assignees", [])) or "Unassigned"
                    lines.append(f"  • '{title}' | Status: {status} | Priority: {prio} | Assignee: {assignees}")

        # Issues Section
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
                    lines.append(f"  • '{title}' | Severity: {sev} | Status: {status}")

        # Project Timeline Section
        if "error" not in project_data and project_data:
            p_name = project_data.get("projectName", "Active Project")
            deadline = project_data.get("deadline") or "Not set"
            days_left = project_data.get("daysRemaining")
            days_str = f"{days_left} days remaining" if days_left is not None else "No deadline set"
            lines.append(f"- **Project Information**: Name: '{p_name}' | Target Deadline: {deadline} ({days_str})")

        # Standup Section
        if need_standup and "error" not in standup_data and standup_data:
            user_tasks = standup_data.get("tasks", [])
            lines.append(f"- **User Daily Standup**: {len(user_tasks)} task(s) assigned to current user.")
            for ut in user_tasks:
                lines.append(f"  • '{ut.get('title', 'Task')}' (Status: {ut.get('status', 'unknown')}, Priority: {ut.get('priority', 'medium')})")

        # Workload Section
        if need_workload and "error" not in workload_data and workload_data:
            members = workload_data.get("members", [])
            lines.append(f"- **Team Workload Breakdown**: {len(members)} team member(s) tracked.")
            for m in members:
                m_name = m.get("name", "Unknown")
                m_role = m.get("role", "member")
                m_tasks = m.get("totalTasks", len(m.get("tasks", [])))
                m_issues = m.get("totalIssues", len(m.get("issues", [])))
                lines.append(f"  • Member: {m_name} ({m_role}) -> Tasks Assigned: {m_tasks} | Issues: {m_issues}")

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
                    eval_llm = ChatOpenAI(
                        model=os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b"),
                        openai_api_key=os.getenv("GROQ_API_KEY", ""),
                        openai_api_base="https://api.groq.com/openai/v1",
                        temperature=0.1,
                    )
                    eval_res = await eval_llm.ainvoke(eval_prompt)
                    lines.append(f"\n### PRD / Document Review & Critical Issues ({filename}):\n{eval_res.content}")
            except Exception as doc_err:
                print(f"[ANALYST WORKER] Document analysis notice: {doc_err}")
                lines.append(f"- **Document Analysis Notice**: {doc_err}")

        findings_text = "\n".join(lines)

        return {
            "_analyst_messages": [RESET_SENTINEL, {"role": "analyst", "content": findings_text}],
            "standup_data": standup_data if need_standup else None,
            "project_insights": project_data,
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

class TaskDraftItem(BaseModel):
    title: str = Field(description="Clean, concise title of the task")
    description: Optional[str] = Field(default="", description="Detailed description or context")
    priority: Optional[str] = Field(default="medium", description="Priority level: 'high', 'medium', or 'low'")


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

        # Check for report scheduler queries
        if "scheduler" in user_query.lower() or "report" in user_query.lower():
            _emit_stream_status(tool_called="get_scheduler", caller="DB Write agent")
            scheduler_info = await fetch_scheduler_async(project_id)
            if scheduler_info:
                exists = scheduler_info.get("exists", False)
                lines.append(f"- **Scheduler Status**: {'Active scheduler configured' if exists else 'No active report scheduler configured'}")

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
                        "     * Populate each task with a clean, concise title, contextual description, and priority ('high', 'medium', 'low').\n"
                        "2. ISSUE CREATION TRIGGER:\n"
                        "   - If the user asks to create issues, log bugs, or says 'create issues of those', 'create issues for these blockers':\n"
                        "     * ALWAYS set is_issue_creation=True.\n"
                        "     * Extract all bugs/issues from the document or previous messages into `issues` with title, description, and severity ('critical', 'high', 'medium', 'low').\n"
                        "3. TITLES & DESCRIPTIONS:\n"
                        "   - Keep task titles concise and actionable (e.g. 'Fix dispute-hold release bug', 'Audit and fix IAM segregation-of-duties violations'). Never return an empty task list if items were discussed or exist in the PRD."
                    )),
                    HumanMessage(content=f"Recent Conversation History:\n{convo_context}{doc_context}\n\nLatest User Query: '{user_query}'"),
                ]
                extracted: DBWriteIntentExtraction = await extractor_llm.ainvoke(extraction_prompt)

                # ── 1. Handle Task Creation HITL Interrupt ──────────────────
                if extracted.is_task_creation and extracted.tasks:
                    task_items_preview = [
                        {
                            "title": t.title.replace("named-", "").strip(),
                            "description": t.description if (t.description and not t.description.startswith("Task requested")) else "",
                            "priority": t.priority if t.priority in ["high", "medium", "low"] else "medium",
                        }
                        for t in extracted.tasks
                    ]
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
                        tasks_for_convex = [
                            {
                                "title": t.get("title", "Task").replace("named-", "").strip(),
                                "description": t.get("description", "") or "",
                                "priority": t.get("priority", "medium"),
                            }
                            for t in final_tasks
                        ]
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
# SUB-AGENT 3: SPRINT WORKER (Sprint Velocity & Lifecycle Management)
# ─────────────────────────────────────────────────────────────────────────────

async def sprint_worker_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Sprint Sub-Agent:
    - Fetches sprint insights, velocity metrics, and timelines concurrently.
    - Handles sprint creation / task assignment workflows.
    - Wrapped in try/except for graceful error reporting.
    """
    _emit_stream_status(status_text="Sprint worker inspecting sprint velocity & backlog...")
    project_id = state.get("project_id") or ""

    if not project_id:
        return {
            "_sprint_messages": [
                RESET_SENTINEL,
                {"role": "sprint", "content": "⚠️ No active project selected. Cannot fetch sprint analytics."},
            ]
        }

    try:
        _emit_stream_status(tool_called="get_sprint_insights", caller="Sprint manager")
        _emit_stream_status(tool_called="get_project_insights", caller="Sprint manager")

        # Fetch sprint insights and project insights in parallel
        sprints_res, project_res = await asyncio.gather(
            fetch_sprint_insights_async(project_id),
            fetch_project_insights_async(project_id),
            return_exceptions=True,
        )

        sprint_data = sprints_res if not isinstance(sprints_res, Exception) else {"error": str(sprints_res)}
        project_data = project_res if not isinstance(project_res, Exception) else {}

        lines = ["### Sprint Sub-Agent Findings:"]
        if "error" in sprint_data:
            lines.append(f"- **Sprint Insights**: ⚠️ {sprint_data['error']}")
        else:
            sprints_list = sprint_data.get("sprints", [])
            active_sprints = [s for s in sprints_list if s.get("status") == "active"]
            completed_sprints = [s for s in sprints_list if s.get("status") == "completed"]
            lines.append(
                f"- **Sprint Analytics**: Total Sprints: {len(sprints_list)} | "
                f"Active: {len(active_sprints)} | Completed: {len(completed_sprints)}"
            )
            if active_sprints:
                curr = active_sprints[0]
                lines.append(f"- **Current Active Sprint**: '{curr.get('name', 'Sprint')}' (Goal: {curr.get('goal', 'N/A')})")

        if project_data and "deadline" in project_data:
            lines.append(f"- **Project Target Deadline**: {project_data.get('deadline')}")

        findings_text = "\n".join(lines)

        return {
            "_sprint_messages": [RESET_SENTINEL, {"role": "sprint", "content": findings_text}],
            "sprint_insights": sprint_data,
        }

    except GraphInterrupt:
        raise
    except Exception as e:
        error_msg = f"Sprint Sub-Agent encountered an error: {e}"
        print(f"[SPRINT WORKER ERROR] {error_msg}")
        return {
            "_sprint_messages": [
                RESET_SENTINEL,
                {"role": "sprint", "content": f"⚠️ [Sprint Data Notice]: {error_msg}"},
            ],
            "active_error": error_msg,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-AGENT 4: MCP INTEGRATIONS WORKER (Slack, Calendly, Linear, Notion, Sentry)
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
        # Execute MCP agent workflow with smart pruning and live token fetching
        mcp_result = await execute_mcp_agent_workflow(
            project_id=project_id,
            user_query=user_query,
            user_name=user_name,
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
    current_date = datetime.now().strftime("%A, %B %d, %Y")

    system_prompt = f"""You are Kaya, an Executive Technical Project Manager on the WEKRAFT Platform.
You are actively managing the project "{project_name}".

CONVERSATION CONTEXT:
- Today's Real-World Date: {current_date}
- Project Name: {project_name}
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
    - Ingests all sub-agent worker outputs (_analyst_messages, _db_write_messages, _sprint_messages, _mcp_messages).
    - Injects project details, deadline awareness, and current date.
    - Synthesizes findings using gpt-4.1-mini, streaming executive PM tokens to user.
    """
    _emit_stream_status(status_text="Kaya is synthesizing executive PM insights...")
    user_name = state.get("user_name") or "there"
    current_date = datetime.now().strftime("%A, %B %d, %Y")

    # Project deadline and name awareness check
    project_insights = state.get("project_insights") or {}
    project_name = project_insights.get("projectName") or state.get("project_name") or "Active Project"
    deadline_val = project_insights.get("deadline")
    days_left = project_insights.get("daysRemaining")

    if deadline_val:
        deadline_text = f"{deadline_val} ({days_left} day(s) remaining from today)"
    else:
        deadline_text = "No deadline currently set"

    # Collect findings from all sub-agents (Internal DB + External MCP integrations)
    findings_blocks = []
    for key, label in [
        ("_analyst_messages", "ANALYST FINDINGS"),
        ("_db_write_messages", "DB WRITE ACTIONS"),
        ("_sprint_messages", "SPRINT FINDINGS"),
        ("_mcp_messages", "THIRD-PARTY MCP INTEGRATION FINDINGS (Sentry, Jira, Vercel, Slack, Linear, Notion, ETC)"),
    ]:
        msgs = state.get(key, [])
        if msgs:
            content = msgs[-1].get("content") if isinstance(msgs[-1], dict) else str(msgs[-1])
            findings_blocks.append(f"--- {label} ---\n{content}")

    findings_prompt = "\n\n".join(findings_blocks) if findings_blocks else "No worker findings generated."

    active_error = state.get("active_error")
    error_instructions = f"\nNote: Active service error notice: {active_error}. Acknowledge this gracefully." if active_error else ""

    system_prompt = f"""You are Kaya, an Executive Technical Project Manager on the WEKRAFT Platform.
You are actively managing the project "{project_name}".

CONVERSATION & TEMPORAL CONTEXT:
- Today's Real-World Date: {current_date}
- Project Name: {project_name}
- Project Target Deadline: {deadline_text}
- User Name: {user_name}
{error_instructions}

YOUR PERSONA & STANDARDS:
1. Executive PM Tone: Speak with crisp, authoritative, supportive professionalism. Avoid generic fluff, filler phrases, or introductory throat-clearing.
2. Temporal Grounding & Proximity: Today is {current_date}. The target deadline is {deadline_text}. Use these real-world temporal anchors to evaluate timeline urgency, overdue risks, and sprint momentum without hallucinating dates.
3. Strict Privacy & Zero Raw Database IDs: NEVER output, mention, or leak internal Convex database IDs (such as 'kn71qtgem...' or 'nd7088...'). Use only the real project name ('{project_name}'), user name ('{user_name}'), and actual task/issue/ticket titles.
4. MANDATORY STRUCTURED MARKDOWN TABLES FOR INTEGRATIONS, ISSUES, TASKS & SPRINTS:
   - CLEAR ITEM COUNTS: Always start each integration or data section with a prominent header indicating the total item count (e.g. `### 🚨 Sentry Issues (Total: 20 Unresolved)` or `### 🚀 Vercel Deployments (Total: 3)` or `### 📋 Jira Tasks (Total: 4)`).
   - COMPREHENSIVE MARKDOWN TABLES: You MUST render ALL retrieved items in clean, formatted Markdown Tables with all relevant details:
     * Sentry: `| ID / Short Key | Issue / Error Title | Culprit / Location | Status | Level / Severity | Events | Users | Link |`
     * Jira / Linear: `| Key | Issue / Task Summary | Status | Priority | Assignee | Link |`
     * Vercel: `| Project | Deployment URL | State / Status | Commit / Branch | Target / Environment | Creator | Link |`
     * Internal Tasks / Issues: `| Task / Issue Title | Status | Priority / Severity | Assignee | Deadline / Environment |`
   - CLICKABLE LINKS: If URLs, permalinks, inspector URLs, or deployment links are available in the worker data, you MUST include clickable Markdown links (e.g. `[View on Sentry](https://...)` or `[Open Deployment](https://...)`) in the Link column.
   - DO NOT COLLAPSE OR TRUNCATE: List every single issue/item in its own table row. Never use placeholder lines like '| ... and 15 more |'.
   - EXECUTIVE SUMMARY AT END: At the very end of your response, provide an authoritative '### 📊 Executive PM Summary & Next Actions' section containing:
     * **Health & Volume Overview**: A crisp breakdown of totals, system stability, and deployment state.
     * **Top 3 Most Critical Items / Blockers**: The top 3 issues that need immediate developer attention (highlighting error frequency, critical priority, or failed deployments).
     * **Recommended Action Plan**: 2-3 concrete next steps for the engineering team.
5. STRICT ZERO-HALLUCINATION & GROUND-TRUTH RULE: Under NO circumstances should you fabricate, simulate, or invent tasks, issues, ticket keys, epics, bug titles, or member assignments that are not present in the SUB-AGENT WORKER DATA. If an integration returns 0 items or returns an error/unauthorized status, explicitly inform the user of that exact status and advise them to reconnect in the Integrations tab. Never invent placeholder tickets.
6. NEVER CLAIM WRITE ACTIONS WITHOUT CONFIRMATION: Never claim or state that tasks, issues, or calendar events 'have been created' or 'have been inserted' unless the SUB-AGENT WORKER DATA explicitly shows a successful write result (e.g. '✅ Bulk created ...'). If no database insertion occurred, do not claim tasks were created.

SUB-AGENT WORKER DATA:
{findings_prompt}
"""

    llm = get_synthesizer_llm()

    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    response = await llm.ainvoke(messages)

    print("\n" + "=" * 70)
    print(f"[KAYA FINAL PM SYNTHESIS RESPONSE]\n{response.content}")
    print("=" * 70 + "\n")

    return {"messages": [response]}



# ─────────────────────────────────────────────────────────────────────────────
# 5. GRAPH COMPILATION
# ─────────────────────────────────────────────────────────────────────────────

def build_kaya_graph():
    """Builds and compiles the master Kaya multi-agent LangGraph workflow."""
    workflow = StateGraph(SupervisorState)

    # Add Nodes (All 4 Sub-Agents + Direct + Synthesizer)
    workflow.add_node("supervisor_router_node", supervisor_router_node)
    workflow.add_node("kaya_direct_node", kaya_direct_node)
    workflow.add_node("analyst_node", analyst_worker_node)
    workflow.add_node("db_write_node", db_write_worker_node)
    workflow.add_node("sprint_node", sprint_worker_node)
    workflow.add_node("mcp_node", mcp_worker_node)
    workflow.add_node("kaya_synthesizer_node", kaya_synthesizer_node)

    # Edge: Start -> Router
    workflow.add_edge(START, "supervisor_router_node")

    # Conditional Fan-Out Edge: Router -> Direct or Parallel Sub-Agents
    workflow.add_conditional_edges(
        "supervisor_router_node",
        route_supervisor,
        ["kaya_direct_node", "analyst_node", "db_write_node", "sprint_node", "mcp_node"],
    )

    # Direct Response terminates directly
    workflow.add_edge("kaya_direct_node", END)

    # Fan-In: All 4 Sub-Agent Workers converge into Kaya Synthesizer
    workflow.add_edge("analyst_node", "kaya_synthesizer_node")
    workflow.add_edge("db_write_node", "kaya_synthesizer_node")
    workflow.add_edge("sprint_node", "kaya_synthesizer_node")
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

