# graph.py — Kaya AI

import os
import operator
from datetime import datetime
from typing import Annotated
import httpx

from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt, Send
from langgraph.checkpoint.memory import InMemorySaver
from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────


_RESET = "__RESET__"


def _reset_or_add(old: list, new: list) -> list:
    """If new starts with the reset sentinel, start fresh. Otherwise append."""
    if new and new[0] == _RESET:
        return new[1:]  # drop sentinel, return fresh list
    return old + new


class KayaState(MessagesState):
    user_id: str
    thread_id: str
    project_id: str | None
    _analyst_messages: Annotated[list, _reset_or_add]  # smart reducer
    _analyst_tool_call_id: str | None


# ─────────────────────────────────────────────────────────────────────────────
# CONVEX READ TOOLS  (executed by analyst_tools node only)
# ─────────────────────────────────────────────────────────────────────────────


@tool
def get_project_tasks(
    project_id: str,
    status: str | None = None,
    priority: str | None = None,
    sprint_id: str | None = None,
) -> dict:
    """Fetch tasks for a project from the database.

    Args:
        project_id: The Convex project ID to query.
        status: Optional filter — one of: 'not started', 'inprogress',
                'reviewing', 'testing', 'completed'.
        priority: Optional filter — one of: 'high', 'medium', 'low'.
        sprint_id: Optional sprint ID to scope results to a specific sprint.

    Returns a list of tasks with: id, title, status, priority, assignedTo
    (names), startDate, endDate, isBlocked, sprintId.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(
        f"[get_project_tasks] querying — project={project_id} status={status} priority={priority}"
    )
    payload: dict = {"projectId": project_id}
    if status:
        payload["status"] = status
    if priority:
        payload["priority"] = priority
    if sprint_id:
        payload["sprintId"] = sprint_id

    try:
        response = httpx.post(f"{convex_url}/getProjectTasks", json=payload, timeout=10)
        response.raise_for_status()
        tasks = response.json().get("tasks", [])
        print(f"[get_project_tasks] ✓ {len(tasks)} tasks returned:")
        for t in tasks:
            print(
                f"  · [{t.get('status')}] {t.get('title')} — priority={t.get('priority')} blocked={t.get('isBlocked')}"
            )
        return {"tasks": tasks, "count": len(tasks)}
    except httpx.HTTPError as e:
        print(f"[get_project_tasks] ✗ ERROR: {e}")
        return {"tasks": [], "count": 0, "error": str(e)}


@tool
def get_project_issues(
    project_id: str,
    status: str | None = None,
    severity: str | None = None,
    environment: str | None = None,
    sprint_id: str | None = None,
) -> dict:
    """Fetch issues for a project from the database.

    Args:
        project_id: The Convex project ID to query.
        status: Optional filter — one of: 'not opened', 'opened',
                'in review', 'reopened', 'closed'.
        severity: Optional filter — one of: 'critical', 'medium', 'low'.
        environment: Optional filter — one of: 'local', 'dev', 'staging',
                     'production'.
        sprint_id: Optional sprint ID to scope results to a specific sprint.

    Returns a list of issues with: id, title, status, severity, environment,
    type, due_date, taskId, assignedTo (names), sprintId.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(
        f"[get_project_issues] querying — project={project_id} status={status} severity={severity} env={environment}"
    )
    payload: dict = {"projectId": project_id}
    if status:
        payload["status"] = status
    if severity:
        payload["severity"] = severity
    if environment:
        payload["environment"] = environment
    if sprint_id:
        payload["sprintId"] = sprint_id

    try:
        response = httpx.post(
            f"{convex_url}/getProjectIssues", json=payload, timeout=10
        )
        response.raise_for_status()
        issues = response.json().get("issues", [])
        print(f"[get_project_issues] ✓ {len(issues)} issues returned:")
        for i in issues:
            print(
                f"  · [{i.get('status')}] {i.get('title')} — severity={i.get('severity')} env={i.get('environment')} assigned={i.get('assignedTo')}"
            )
        return {"issues": issues, "count": len(issues)}
    except httpx.HTTPError as e:
        print(f"[get_project_issues] ✗ ERROR: {e}")
        return {"issues": [], "count": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DESCRIPTORS  (LLM schema only — intercepted by assign_tool)
# ─────────────────────────────────────────────────────────────────────────────


@tool
def create_calendar_event(
    project_id: str,
    title: str,
    description: str,
    event_type: str,
    start_iso: str,
    end_iso: str,
) -> str:
    """Create a calendar event for a project."""
    return "intercepted"


@tool
def ask_project_analyst(query: str, project_id: str) -> str:
    """Delegate a data question to the Project Analyst subagent.

    The analyst has read access to the project's tasks and issues and can
    answer questions like:
    - "What tasks are blocked?"
    - "Show me all critical issues in production"
    - "Which high-priority tasks are still not started?"
    - "Who is assigned to the most open issues?"
    - "Summarise sprint progress"

    Args:
        query: Natural language question about the project's tasks or issues.
        project_id: The project to query — always pass the active project_id.
    """
    return "intercepted"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Convex write (only after HITL approval)
# ─────────────────────────────────────────────────────────────────────────────


async def write_calendar_event_to_convex(payload: dict) -> str:
    convex_url = os.getenv("CONVEX_SITE_URL")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{convex_url}/createCalendarEvent", json=payload, timeout=10
            )
            response.raise_for_status()
            result = response.json()
            return (
                f"✅ Calendar event created: '{payload['title']}' "
                f"(id: {result.get('id', 'unknown')})"
            )
    except httpx.HTTPError as e:
        return f"❌ Failed to create calendar event: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# LLM MODELS
# ─────────────────────────────────────────────────────────────────────────────

_kaya_llm = ChatOpenAI(
    model=os.getenv("KAYA_MODEL", "gpt-4.1-mini"),
    temperature=0.3,
    streaming=True,
).bind_tools([create_calendar_event, ask_project_analyst])

# Analyst LLM — bound to its own read tools only, zero temperature for factual accuracy
_analyst_llm = ChatOpenAI(
    model=os.getenv("ANALYST_MODEL", "gpt-4.1-nano"),
    temperature=0,
    streaming=True,
).bind_tools([get_project_tasks, get_project_issues])

_mem0 = MemoryClient()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

KAYA_SYSTEM = """You are Kaya, an AI Product Manager agent.
You help product teams clarify requirements, write PRDs, break down features into tasks,
prioritize backlogs, and surface insights from project data.

Be concise, opinionated, and practical. Ask clarifying questions when needed.
Always think from the user's perspective and business impact.

When answering questions about tasks, issues, blockers, or sprint progress:
- Delegate to ask_project_analyst with a clear natural-language query and the active project_id.
- Wait for the analyst's findings, then synthesise them into a helpful PM-level response.

When creating calendar events:
- Use the create_calendar_event tool (no need to ask for confirmation).
- make sure to check type (event or milestone).
- Use ISO 8601 format for dates (e.g. 2025-04-22T00:00:00).
-if Tool call failed , try again with right parameters."""

_ANALYST_SYSTEM = """You are the Project Analyst — a specialist with read-only access to project data.

Your job:
1. Use get_project_tasks and get_project_issues to fetch data needed to answer the query.
2. IMPORTANT: Tools filter server-side. For MULTIPLE statuses make one call per status:
   - "completed AND not started tasks" → call 1: status="completed", call 2: status="not started"
3. To get ALL tasks with no filter, call with only project_id and no status.
4. Make as many tool calls as needed — never stop after one call if the query needs more.
5. Produce a clear, structured final answer with bullet points and counts.

Rules:
- Always pass project_id — never fetch without it.
- Be concise and factual. No fluff.
- If data is empty for a status, say so explicitly.
- Always return the whole data , not half information."""


# ─────────────────────────────────────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────────────────────────────────────


def kaya(state: KayaState) -> dict:
    user_id = state["user_id"]
    project_id = state.get("project_id")
    messages = state["messages"]

    last_user_msg = next(
        (m.content for m in reversed(messages) if m.type == "human"), ""
    )

    recalled = _mem0.search(last_user_msg, filters={"user_id": user_id})
    memory_block = (
        "\n\nRelevant context from past sessions:\n"
        + "\n".join(f"- {m['memory']}" for m in recalled.get("results", []))
        if recalled.get("results")
        else ""
    )
    project_context = (
        f"\n\nActive project_id: {project_id}\n"
        "Always pass this project_id when calling any project tool."
        if project_id
        else ""
    )

    full_messages = [
        SystemMessage(content=KAYA_SYSTEM + memory_block + project_context)
    ] + messages
    response = _kaya_llm.invoke(full_messages)

    # Save to memory only on plain conversation turns (no tool calls)
    if not response.tool_calls and last_user_msg:
        try:
            _mem0.add(
                [
                    {"role": "user", "content": last_user_msg},
                    {"role": "assistant", "content": response.content},
                ],
                user_id=user_id,
            )
        except Exception as e:
            print(f"[mem0] failed: {e}")

    return {"messages": [response]}


def project_analyst(tool_call: dict) -> dict:
    args = tool_call["args"]
    query = args.get("query", "")
    project_id = args.get("project_id", "")
    print(f"[project_analyst] Seeding — query='{query}' project={project_id}")

    return {
        "_analyst_tool_call_id": tool_call["id"],
        "_analyst_messages": [
            _RESET,  # ← tells reducer to wipe old messages
            {"role": "system", "content": _ANALYST_SYSTEM},
            {"role": "user", "content": f"Project ID: {project_id}\nQuestion: {query}"},
        ],
    }


# analyst_think — just appends, no change needed
def analyst_think(state: KayaState) -> dict:
    analyst_messages = state.get("_analyst_messages", [])
    print(f"[analyst_think] Invoking with {len(analyst_messages)} messages")

    response = _analyst_llm.invoke(analyst_messages)

    serialized_tool_calls = [
        {"id": tc["id"], "name": tc["name"], "args": tc["args"], "type": "tool_call"}
        for tc in (response.tool_calls or [])
    ]

    return {
        "_analyst_messages": [
            {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": serialized_tool_calls,
            }
        ]
    }


def analyst_tools(tool_call: dict) -> dict:
    name = tool_call["name"]
    args = tool_call["args"]
    print(f"[analyst_tools] Executing — tool={name}")

    if name == "get_project_tasks":
        result = get_project_tasks.invoke(args)
    elif name == "get_project_issues":
        result = get_project_issues.invoke(args)
    else:
        result = {"error": f"Unknown tool: {name}"}

    return {
        "_analyst_messages": [
            {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call["id"],
                "name": name,
            }
        ]
    }


def analyst_done(closing_msg: ToolMessage) -> dict:
    """
    Receives the closing ToolMessage from analyst_route.
    Appends it to kaya's messages so kaya can synthesise the final answer.
    """
    return {"messages": [closing_msg]}


async def tools(tool_call: dict) -> dict:
    """HITL node — pauses for user approval before writing the calendar event."""
    args = tool_call["args"]

    start_ms = int(datetime.fromisoformat(args["start_iso"]).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(args["end_iso"]).timestamp() * 1000)

    event_payload = {
        "projectId": args["project_id"],
        "title": args["title"],
        "description": args["description"],
        "type": args["event_type"],
        "start": start_ms,
        "end": end_ms,
        "allDay": True,
    }

    decision = interrupt(
        {
            "tool": "create_calendar_event",
            "message": "Review this calendar event before it's saved.",
            "preview": {
                "title": args["title"],
                "description": args["description"],
                "type": args["event_type"],
                "start": args["start_iso"],
                "end": args["end_iso"],
                "allDay": True,
            },
        }
    )

    if not isinstance(decision, dict) or decision.get("action") != "approve":
        return {
            "messages": [
                ToolMessage(
                    content="❌ Calendar event creation cancelled by user.",
                    tool_call_id=tool_call["id"],
                    name="create_calendar_event",
                )
            ]
        }

    result_msg = await write_calendar_event_to_convex(event_payload)
    return {
        "messages": [
            ToolMessage(
                content=result_msg,
                tool_call_id=tool_call["id"],
                name="create_calendar_event",
            )
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────────────────────


def assign_tool(state: KayaState):
    """Routes Kaya's tool calls to the correct node."""
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END

    sends = []
    for tc in last_message.tool_calls:
        match tc["name"]:
            case "create_calendar_event":
                sends.append(Send("tools", tc))
            case "ask_project_analyst":
                sends.append(Send("project_analyst", tc))

    return sends if sends else END


def analyst_route(state: KayaState):
    """
    After analyst_think:
    - Last message has tool_calls → fan out to analyst_tools (one Send per call)
    - Last message has no tool_calls → analyst is done → close loop to kaya
    """
    analyst_messages = state.get("_analyst_messages", [])
    last = analyst_messages[-1] if analyst_messages else None

    if last and last.get("tool_calls"):
        return [Send("analyst_tools", tc) for tc in last["tool_calls"]]

    analyst_tool_call_id = state.get("_analyst_tool_call_id", "unknown")
    final_content = last.get("content", "No findings.") if last else "No findings."
    print(f"[analyst_route] Analyst done. Closing tool_call_id={analyst_tool_call_id}")

    return Send(
        "analyst_done",
        ToolMessage(
            content=final_content,
            tool_call_id=analyst_tool_call_id,
            name="ask_project_analyst",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

checkpointer = InMemorySaver()


def build_graph():
    g = StateGraph(KayaState)

    # Core nodes
    g.add_node("kaya", kaya)
    g.add_node("tools", tools)

    # Analyst subgraph nodes (flattened — no create_react_agent)
    g.add_node("project_analyst", project_analyst)  # seeds _analyst_messages
    g.add_node("analyst_think", analyst_think)  # LLM decides next step
    g.add_node("analyst_tools", analyst_tools)  # executes one tool call
    g.add_node("analyst_done", analyst_done)  # closes loop → kaya

    # Kaya flow
    g.add_edge(START, "kaya")
    g.add_conditional_edges("kaya", assign_tool)
    g.add_edge("tools", "kaya")

    # Analyst flow
    g.add_edge("project_analyst", "analyst_think")  # seed → think
    g.add_conditional_edges("analyst_think", analyst_route)  # think → tools | done
    g.add_edge("analyst_tools", "analyst_think")  # tool result → think (loop)
    g.add_edge("analyst_done", "kaya")  # findings → kaya synthesises

    return g.compile(checkpointer=checkpointer)


graph = build_graph()
print("[graph] Kaya agent graph compiled successfully")
