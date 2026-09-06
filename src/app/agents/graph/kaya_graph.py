"""
kaya_graph.py — Master Multi-Agent LangGraph for Kaya AI Product Manager.

Architecture:
1. Supervisor Router (Groq Fast Structured Intent Classifier)
2. Conditional Parallel Fan-Out:
   - 'direct_response' -> Kaya Direct Response (Groq 120b / fast streaming)
   - ['analyst', 'db_write', 'sprint'] -> Parallel Sub-Agent Worker execution
3. 3 Specialized Sub-Agents:
   - Analyst Sub-Agent (Parallel read tools: standups, tasks, issues, workloads, project timeline)
   - DB Write Sub-Agent (Mem0 memory search, calendar events, report scheduler, PRD bulk tools)
   - Sprint Sub-Agent (Sprint analytics/velocity, sprint creation, task allocation)
4. Fan-In Aggregation & Unified Synthesis:
   - Kaya Synthesizer (gpt-4.1-mini): Ingests all worker findings + project deadline awareness,
     streams final PM-level tokens to user.
5. Zero Cascading Failures:
   - Isolated try/except in every worker returning structured error envelopes to Kaya.
"""

import os
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
    search_user_memory,
    write_calendar_event_to_convex,
    write_sprint_to_convex,
    write_items_to_sprint,
    write_scheduler_to_convex,
    write_bulk_tasks_to_convex,
    write_bulk_issues_to_convex,
)

load_dotenv(override=True)



# ─────────────────────────────────────────────────────────────────────────────
# 1. SUPERVISOR ROUTER DECISION SCHEMA & PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    actions: List[str] = Field(
        description=(
            "List of sub-agent actions to trigger in parallel (can select 1 or multiple):\n"
            "- 'db_write': user memory search (Mem0), calendar events, report scheduler setup, or bulk task creation\n"
            "- 'analyst': task/issue summaries, daily standup, member workloads, project insights/deadlines\n"
            "- 'sprint': sprint insights/velocity, sprint creation, or backlog item assignments\n"
            "- 'direct_response': simple greeting, casual conversation, or general product manager chat"
        )
    )
    reasoning: str = Field(description="Clear reasoning for the routing decision.")


ROUTER_SYSTEM_PROMPT = """You are the primary Supervisor Router for the WEKRAFT AI Platform.
Analyze the user's incoming query and decide which specialized sub-agent workers to trigger:

Available Actions:
1. 'db_write': State mutations or personal memory: Long-term memory search (Mem0), calendar event scheduling, report scheduler setup, or bulk PRD task/issue generation.
2. 'analyst': Read-only analytics: User daily standup, task summaries, issue tracking, member workloads, project health, or project deadlines/timelines.
3. 'sprint': Sprint velocity, active sprint progress, sprint creation, or backlog item allocation.
4. 'direct_response': Greetings (hi, hello), casual conversation, or general questions requiring no live database queries.

Rules:
- Select MULTIPLE sub-agents if the user query asks for multiple aspects (e.g. 'Show my standup and sprint velocity' -> ['analyst', 'sprint']).
- If the query is just a greeting or general dialogue, choose ONLY ['direct_response'].
"""


async def route_user_request(user_input: str, project_id: Optional[str] = None) -> SupervisorDecision:
    """Evaluates user input using Groq LLM router and outputs structured SupervisorDecision."""
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    router_model = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b")

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"User Query: '{user_input}' | Active Project ID: {project_id or 'none'}"),
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
        safe_reasoning = decision.reasoning.encode("ascii", "replace").decode("ascii")
        print(f"[ROUTER DECISION] Actions: {decision.actions} | Reasoning: {safe_reasoning}")
        return decision

    except Exception as e:
        safe_err = str(e).encode("ascii", "replace").decode("ascii")
        print(f"[ROUTER WARNING] Fallback routing due to: {safe_err}")
        return SupervisorDecision(
            actions=["direct_response"],
            reasoning=f"Fallback routing due to LLM error: {safe_err}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _get_latest_user_text(messages: List[BaseMessage]) -> str:
    """Extracts text content of the latest human message."""
    for m in reversed(messages):
        if m.type == "human":
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _emit_stream_status(status_text: str):
    """Emits custom SSE progress event for frontend status indicator."""
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        writer({"agent_status": status_text})
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────

async def supervisor_router_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Node: Evaluates user query and determines parallel execution branches."""
    messages = state.get("messages", [])
    user_query = _get_latest_user_text(messages)
    project_id = state.get("project_id")

    _emit_stream_status("Kaya is understanding your request...")
    decision = await route_user_request(user_query, project_id)

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
    _emit_stream_status("Analyst worker analyzing project health & tasks in parallel...")
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

        findings_text = "\n".join(lines)

        return {
            "_analyst_messages": [RESET_SENTINEL, {"role": "analyst", "content": findings_text}],
            "standup_data": standup_data if need_standup else None,
            "project_insights": project_data,
        }

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

async def db_write_worker_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    DB Write Sub-Agent:
    - Handles Mem0 long-term memory searches.
    - Handles state mutations and HITL write flows (calendar events, report schedulers, bulk PRD items).
    - Wrapped in try/except for graceful error reporting.
    """
    _emit_stream_status("DB Write worker searching memory & preparing actions...")
    project_id = state.get("project_id") or ""
    user_id = state.get("user_id") or ""
    messages = state.get("messages", [])
    user_query = _get_latest_user_text(messages)

    try:
        memories = []
        # Search Mem0 for user memory/preferences
        if any(k in user_query.lower() for k in ["remember", "preference", "last time", "memory", "favorite", "my info"]):
            mem_result = search_user_memory.invoke({"user_id": user_id, "query": user_query})
            memories = mem_result.get("memories", [])

        # Check for report scheduler queries
        scheduler_info = None
        if "scheduler" in user_query.lower() or "report" in user_query.lower():
            scheduler_info = await fetch_scheduler_async(project_id)

        lines = ["### DB Write / Memory Sub-Agent Findings:"]
        if memories:
            lines.append(f"- **Retrieved Memory**: Found {len(memories)} relevant past user context entries.")
        if scheduler_info:
            exists = scheduler_info.get("exists", False)
            lines.append(f"- **Scheduler Status**: {'Active scheduler configured' if exists else 'No active report scheduler configured'}")
        if not memories and not scheduler_info:
            lines.append("- **DB Write Status**: Ready for write/mutation commands.")

        summary_text = "\n".join(lines)

        return {
            "_db_write_messages": [RESET_SENTINEL, {"role": "db_write", "content": summary_text}],
            "retrieved_memory": [m.get("memory", "") for m in memories if isinstance(m, dict)],
        }

    except Exception as e:
        error_msg = f"DB Write Sub-Agent encountered an error: {e}"
        print(f"[DB WRITE WORKER ERROR] {error_msg}")
        return {
            "_db_write_messages": [
                RESET_SENTINEL,
                {"role": "db_write", "content": f"⚠️ [Write/Memory Notice]: {error_msg}"},
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
    _emit_stream_status("Sprint worker inspecting sprint velocity & backlog...")
    project_id = state.get("project_id") or ""

    if not project_id:
        return {
            "_sprint_messages": [
                RESET_SENTINEL,
                {"role": "sprint", "content": "⚠️ No active project selected. Cannot fetch sprint analytics."},
            ]
        }

    try:
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


def get_chat_llm(model_type: str = "fast") -> ChatOpenAI:
    """Returns ChatOpenAI instance configured for OpenAI or Groq based on available keys with max_tokens set."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    max_tokens = int(os.getenv("KAYA_MAX_TOKENS", "700"))

    if openai_key and model_type == "deep":
        model_name = os.getenv("KAYA_DEEP_MODEL", "gpt-4.1-mini")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=openai_key,
            temperature=0.2,
            max_tokens=max_tokens,
            streaming=True,
        )
    elif groq_key:
        groq_model = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
        return ChatOpenAI(
            model=groq_model,
            openai_api_key=groq_key,
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.2,
            max_tokens=max_tokens,
            streaming=True,
        )
    else:
        model_name = os.getenv("KAYA_FAST_MODEL", "gpt-4.1-mini")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=openai_key or "none",
            temperature=0.2,
            max_tokens=max_tokens,
            streaming=True,
        )



async def kaya_direct_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """Kaya Direct Response Node: Fast conversational greeting/chat with token streaming."""
    _emit_stream_status("Kaya is typing...")
    user_name = state.get("user_name") or "there"
    project_id = state.get("project_id")
    current_date = datetime.now().strftime("%B %d, %Y")

    system_prompt = (
        f"{KAYA_SYSTEM_BASE}\n"
        f"You are talking directly to {user_name}.\n"
        f"Current Date: {current_date}\n"
        f"Active Project ID: {project_id or 'None'}\n"
        "Provide a warm, concise, and professional Product Manager greeting or answer."
    )

    llm = get_chat_llm(model_type="fast")

    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    response = await llm.ainvoke(messages)

    return {"messages": [response]}


async def kaya_synthesizer_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Kaya Synthesizer Node:
    - Ingests all sub-agent worker outputs (_analyst_messages, _db_write_messages, _sprint_messages).
    - Injects project details, deadline awareness, and current date.
    - Synthesizes findings into a unified PM response, streaming tokens to user.
    """
    _emit_stream_status("Kaya is synthesizing executive PM insights...")
    user_name = state.get("user_name") or "there"
    project_id = state.get("project_id") or "Not set"
    current_date = datetime.now().strftime("%B %d, %Y")

    # Project deadline awareness check
    project_insights = state.get("project_insights") or {}
    deadline_text = f"Project Deadline: {project_insights.get('deadline', 'N/A')} ({project_insights.get('daysRemaining', 'N/A')} days remaining)"

    # Collect findings from sub-agents
    findings_blocks = []
    for key, label in [
        ("_analyst_messages", "ANALYST FINDINGS"),
        ("_db_write_messages", "MEMORY & WRITE ACTIONS"),
        ("_sprint_messages", "SPRINT FINDINGS"),
    ]:
        msgs = state.get(key, [])
        if msgs:
            content = msgs[-1].get("content") if isinstance(msgs[-1], dict) else str(msgs[-1])
            findings_blocks.append(f"--- {label} ---\n{content}")

    findings_prompt = "\n\n".join(findings_blocks) if findings_blocks else "No worker findings generated."

    active_error = state.get("active_error")
    error_instructions = f"\nNote: Active service error detected: {active_error}. Acknowledge this gracefully." if active_error else ""

    system_prompt = f"""{KAYA_SYSTEM_BASE}

CONVERSATION CONTEXT:
- Talking to: {user_name}
- Active Project ID: {project_id}
- Today's Date: {current_date}
- {deadline_text}
{error_instructions}

SUB-AGENT WORKER DATA (DO NOT expose raw labels to user, synthesize seamlessly):
{findings_prompt}

INSTRUCTIONS:
- Synthesize all findings into a crisp, PM-level answer for the user.
- Emphasize key numbers, active blockers, sprint velocity, and upcoming deadlines.
- Keep formatting clean with structured Markdown.
"""

    model_type = config.get("configurable", {}).get("model", "fast") if config else "fast"
    llm = get_chat_llm(model_type=model_type)

    messages = [SystemMessage(content=system_prompt)] + list(state.get("messages", []))
    response = await llm.ainvoke(messages)

    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────────────────────
# 5. GRAPH COMPILATION
# ─────────────────────────────────────────────────────────────────────────────

def build_kaya_graph():
    """Builds and compiles the master Kaya multi-agent LangGraph workflow."""
    workflow = StateGraph(SupervisorState)

    # Add Nodes
    workflow.add_node("supervisor_router_node", supervisor_router_node)
    workflow.add_node("kaya_direct_node", kaya_direct_node)
    workflow.add_node("analyst_node", analyst_worker_node)
    workflow.add_node("db_write_node", db_write_worker_node)
    workflow.add_node("sprint_node", sprint_worker_node)
    workflow.add_node("kaya_synthesizer_node", kaya_synthesizer_node)

    # Edge: Start -> Router
    workflow.add_edge(START, "supervisor_router_node")

    # Conditional Fan-Out Edge: Router -> Direct or Parallel Sub-Agents
    workflow.add_conditional_edges(
        "supervisor_router_node",
        route_supervisor,
        ["kaya_direct_node", "analyst_node", "db_write_node", "sprint_node"],
    )

    # Direct Response terminates directly
    workflow.add_edge("kaya_direct_node", END)

    # Fan-In: All Sub-Agent Workers converge into Kaya Synthesizer
    workflow.add_edge("analyst_node", "kaya_synthesizer_node")
    workflow.add_edge("db_write_node", "kaya_synthesizer_node")
    workflow.add_edge("sprint_node", "kaya_synthesizer_node")

    # Synthesizer terminates at END
    workflow.add_edge("kaya_synthesizer_node", END)

    # Checkpointer for durable execution & HITL resumes
    checkpointer = get_checkpointer()
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    print("[KAYA GRAPH] Master Supervisor & Multi-Agent Graph compiled successfully.")
    return compiled_graph


# Expose singleton compiled graph
kaya_graph = build_kaya_graph()
