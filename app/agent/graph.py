# graph.py — Kaya AI

import os
from datetime import datetime
import httpx

from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt, Send
from langgraph.checkpoint.memory import InMemorySaver
from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv()

CONVEX_SITE_URL = os.getenv("CONVEX_SITE_URL")


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
class KayaState(MessagesState):
    user_id: str
    thread_id: str
    project_id: str | None


# ─────────────────────────────────────────────────────────────────────────────
# CONVEX READ TOOLS  (used by project_analyst subagent only)
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
    print(
        f"[get_project_tasks] project={project_id} status={status} priority={priority}"
    )
    payload: dict = {"projectId": project_id}
    if status:
        payload["status"] = status
    if priority:
        payload["priority"] = priority
    if sprint_id:
        payload["sprintId"] = sprint_id

    try:
        response = httpx.post(
            f"{CONVEX_SITE_URL}/getProjectTasks",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        tasks = data.get("tasks", [])
        print(f"[get_project_tasks] ✓ {len(tasks)} tasks returned")
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
    print(
        f"[get_project_issues] project={project_id} status={status} severity={severity}"
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
            f"{CONVEX_SITE_URL}/getProjectIssues",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        issues = data.get("issues", [])
        print(f"[get_project_issues] ✓ {len(issues)} issues returned")
        return {"issues": issues, "count": len(issues)}
    except httpx.HTTPError as e:
        print(f"[get_project_issues] ✗ ERROR: {e}")
        return {"issues": [], "count": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DESCRIPTOR (LLM schema only)
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
# HELPER: Convex write (only after approval)
# ─────────────────────────────────────────────────────────────────────────────
async def write_calendar_event_to_convex(payload: dict) -> str:
    CONVEX_SITE_URL = os.getenv("CONVEX_SITE_URL")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CONVEX_SITE_URL}/createCalendarEvent",
                json=payload,
                timeout=10,
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
# LLM + MEMORY
# ─────────────────────────────────────────────────────────────────────────────
_kaya_llm = ChatOpenAI(
    model=os.getenv("KAYA_MODEL", "gpt-4.1-mini"),
    temperature=0.3,
    streaming=True,
).bind_tools([create_calendar_event, ask_project_analyst])

# Project analyst — fast, precise, zero hallucination tolerance
_analyst_llm = ChatOpenAI(
    model=os.getenv("ANALYST_MODEL", "gpt-4.1-nano"),
    temperature=0,
    streaming=True,
)

_mem0 = MemoryClient()

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT ANALYST SUBAGENT  (ReAct loop with read-only Convex tools)
# ─────────────────────────────────────────────────────────────────────────────

_analyst_prompt = """You are the Project Analyst — a specialist subagent with read-only access to
project data (tasks and issues).
 
Your job:
1. Use get_project_tasks and get_project_issues to fetch the data needed to answer the query.
2. Apply filters (status, priority, severity, environment) to narrow results — don't fetch everything if a filter applies.
3. Analyse the returned data and produce a clear, structured answer.
 
Rules:
- Always filter by project_id — never fetch data without it.
- Use multiple tool calls if needed (e.g. fetch blocked tasks AND critical issues separately).
- Be concise and factual. No fluff. Bullet points for lists, numbers for counts.
- If the data is empty, say so clearly.
- Never guess — only report what the data shows."""

project_analyst_subagent = create_react_agent(
    model=_analyst_llm,
    tools=[get_project_tasks, get_project_issues],
    name="project_analyst",
    prompt=_analyst_prompt,
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
KAYA_SYSTEM = """You are Kaya, an AI Product Manager agent.
You help product teams clarify requirements, write PRDs, break down features into tasks,
prioritize backlogs.

Be concise, opinionated, and practical. Ask clarifying questions when needed.
Always think from the user's perspective and business impact.

When creating calendar events:
- Use the create_calendar_event tool.(no need to ask for confirmation).
- All events are all-day by default.
- Use ISO 8601 format for dates (e.g. 2025-04-22T00:00:00)."""


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
        f"\n\nActive project_id: {project_id}\nAlways pass this project_id when calling any project tool."
        if project_id
        else ""
    )

    system_content = KAYA_SYSTEM + memory_block + project_context
    full_messages = [SystemMessage(content=system_content)] + messages

    response = _kaya_llm.invoke(full_messages)

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


async def project_analyst(tool_call: dict) -> dict:
    """Runs the project analyst ReAct subagent and returns findings as a ToolMessage."""
    args = tool_call["args"]
    query = args.get("query", "")
    project_id = args.get("project_id", "")

    print(f"[project_analyst] Starting — query='{query}' project={project_id}")

    result = await project_analyst_subagent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(f"Project ID: {project_id}\n" f"Question: {query}")
                )
            ]
        }
    )

    findings = result["messages"][-1].content
    print(f"[project_analyst] Done — {len(findings)} chars returned")

    return {
        "messages": [
            ToolMessage(
                content=findings,
                tool_call_id=tool_call["id"],
                name="ask_project_analyst",
            )
        ]
    }


async def tools(tool_call: dict) -> dict:
    """HITL node for calendar event (named 'tools' so your UI works)"""
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
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END

    sends = []
    for tool_call in last_message.tool_calls:
        match tool_call["name"]:
            case "create_calendar_event":
                sends.append(Send("tools", tool_call))
            case "ask_project_analyst":
                sends.append(Send("project_analyst", tool_call))

    return sends if sends else END


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

checkpointer = InMemorySaver()


def build_graph():
    g = StateGraph(KayaState)

    g.add_node("kaya", kaya)
    g.add_node("tools", tools)
    g.add_node("project_analyst", project_analyst)

    g.add_edge(START, "kaya")
    g.add_conditional_edges("kaya", assign_tool)
    g.add_edge("tools", "kaya")  # after calendar event → back to Kaya
    g.add_edge("project_analyst", "kaya")  # after data insight → back to Kaya

    return g.compile(checkpointer=checkpointer)


graph = build_graph()
print("[graph] Kaya agent graph compiled successfully")
