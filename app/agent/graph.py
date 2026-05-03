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
from langchain_core.runnables import RunnableConfig
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
    user_name: str | None
    thread_id: str
    project_id: str | None
    _analyst_messages: Annotated[list, _reset_or_add]  # smart reducer
    _analyst_tool_call_id: str | None


# ─────────────────────────────────────────────────────────────────────────────
# CONVEX READ TOOLS  (executed by analyst_tools node only)
# ─────────────────────────────────────────────────────────────────────────────


@tool
def get_tasks_summary(project_id: str) -> dict:
    """Fetch a high-level summary of all tasks including critical and active ones.
    Useful for getting a quick overview of project health and identifying bottlenecks.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(f"[get_tasks_summary] querying — project={project_id}")
    try:
        response = httpx.post(
            f"{convex_url}/getTasksSummary", json={"projectId": project_id}, timeout=10
        )
        response.raise_for_status()
        summary = response.json().get("tasksSummary", {})
        print(f"[get_tasks_summary] ✓ returned")
        return summary
    except httpx.HTTPError as e:
        print(f"[get_tasks_summary] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_issues_summary(project_id: str) -> dict:
    """Fetch a summary of all issues, focusing on active and critical ones.
    Useful for identifying major blockers and critical bugs.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(f"[get_issues_summary] querying — project={project_id}")
    try:
        response = httpx.post(
            f"{convex_url}/getIssuesSummary", json={"projectId": project_id}, timeout=10
        )
        response.raise_for_status()
        summary = response.json().get("issuesSummary", {})
        print(f"[get_issues_summary] ✓ returned")
        return summary
    except httpx.HTTPError as e:
        print(f"[get_issues_summary] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_member_workload(project_id: str) -> dict:
    """Returns a detailed breakdown of each team member's current task and issue assignments.
    Useful for load balancing and seeing who is busy.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(f"[get_member_workload] querying — project={project_id}")
    try:
        response = httpx.post(
            f"{convex_url}/getMemberWorkloadPYAgent",
            json={"projectId": project_id},
            timeout=10,
        )
        response.raise_for_status()
        members = response.json().get("members", [])
        print(f"[get_member_workload] ✓ {len(members)} members returned")
        return {"members": members}
    except httpx.HTTPError as e:
        print(f"[get_member_workload] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_user_standup(project_id: str, user_id: str) -> dict:
    """Fetch active tasks and open issues assigned to a specific user.
    Useful for daily standups and helping the user prioritize their work for today and tomorrow.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(f"[get_user_standup] querying — project={project_id} user={user_id}")
    try:
        response = httpx.post(
            f"{convex_url}/getUserStandup",
            json={"projectId": project_id, "userId": user_id},
            timeout=10,
        )
        response.raise_for_status()
        standup = response.json().get("standup", {})
        print(f"[get_user_standup] ✓ returned")
        return standup
    except httpx.HTTPError as e:
        print(f"[get_user_standup] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_sprint_insights(project_id: str) -> dict:
    """Fetch comprehensive analytics for all project sprints, including progress metrics and timelines.
    Useful for understanding sprint velocity and overall progress.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(f"[get_sprint_insights] querying — project={project_id}")
    try:
        response = httpx.post(
            f"{convex_url}/getSprintInsights",
            json={"projectId": project_id},
            timeout=10,
        )
        response.raise_for_status()
        sprints = response.json().get("sprints", [])
        print(f"[get_sprint_insights] ✓ {len(sprints)} sprints returned")
        return {"sprints": sprints}
    except httpx.HTTPError as e:
        print(f"[get_sprint_insights] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_project_insights(project_id: str) -> dict:
    """Fetch basic project timeline information like deadline and days remaining.
    Useful for overall project status and tracking against the final deadline.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(f"[get_project_insights] querying — project={project_id}")
    try:
        response = httpx.post(
            f"{convex_url}/getProjectInsights",
            json={"projectId": project_id},
            timeout=10,
        )
        response.raise_for_status()
        insights = response.json().get("projectInsights", {})
        print(f"[get_project_insights] ✓ returned")
        return insights
    except httpx.HTTPError as e:
        print(f"[get_project_insights] ✗ ERROR: {e}")
        return {"error": str(e)}



@tool
def get_scheduler(project_id: str) -> dict:
    """Fetch the current report scheduler for a project.

    Returns scheduler details if one exists:
        name: scheduler label
        frequencyDays: how often the report runs (minimum 3 days)
        recipientEmail: the email address where reports are sent
        isActive: whether it is currently active
        lastRunAt: unix ms of last run, or null
        nextRunAt: unix ms of scheduled next run

    Returns {"exists": false} if no scheduler has been set up yet.

    Args:
        project_id: The Convex project ID to query.
    """
    convex_url = os.getenv("CONVEX_SITE_URL")
    print(f"[get_scheduler] querying — project={project_id}")
    try:
        response = httpx.post(
            f"{convex_url}/getScheduler",
            json={"projectId": project_id},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        scheduler = data.get("scheduler")
        if not scheduler:
            print("[get_scheduler] ✓ no scheduler found")
            return {"exists": False}
        print(
            f"[get_scheduler] ✓ name={scheduler.get('name')} "
            f"freq={scheduler.get('frequencyDays')}d "
            f"email={scheduler.get('recipientEmail')} "
            f"active={scheduler.get('isActive')}"
        )
        return {"exists": True, **scheduler}
    except httpx.HTTPError as e:
        print(f"[get_scheduler] ✗ ERROR: {e}")
        return {"exists": False, "error": str(e)}


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


@tool
def create_sprint(
    project_id: str,
    sprint_name: str,
    sprint_goal: str,
    start_date: str,
    end_date: str,
) -> str:
    """Create a new sprint for the project.

    Call this ONLY after:
    1. ask_project_analyst has confirmed Project remaining days and available Task counts.
    2. The user has provided sprint_name, sprint_goal, start_date and end_date (duration of the sprint)

    duration_days must not exceed the project's remaining days.

    Args:
        project_id: The active project ID.
        sprint_name: Unique name for the sprint (e.g. 'sprint-auth').
        sprint_goal: What this sprint aims to achieve.
        startDate: Start date of the sprint (YYYY-MM-DD).
        endDate: End date of the sprint (YYYY-MM-DD).
    """
    return "intercepted"


@tool
def add_items_to_sprint(sprint_id: str) -> str:
    """Trigger the item selection UI so the user can pick tasks for the sprint.

    Call immediately after create_sprint succeeds.
    The UI shows all available tasks — user selects and confirms.
    You do NOT need task IDs — the UI and graph handle that.

    Args:
        sprint_id: The ID returned by create_sprint.
    """
    return "intercepted"


@tool
def setup_report_scheduler(project_id: str) -> str:
    """Open the scheduler setup form for the user to configure automated reports.

    Call this when the user wants to:
    - Set up automated / scheduled reports for a project
    - Change how often reports are generated
    - Enable or disable an existing scheduler
    - Set or update the recipient email for reports (optional, defaults to project owner)

    The UI will show a form — no need to ask the user for values upfront.
    Once the user submits the form, the scheduler will be created or updated automatically.

    Args:
        project_id: The active project ID.
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


async def write_sprint_to_convex(payload: dict) -> dict:
    convex_url = os.getenv("CONVEX_SITE_URL")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{convex_url}/createSprint", json=payload, timeout=10
        )
        response.raise_for_status()
        return response.json()


async def write_items_to_sprint(sprint_id: str, task_ids: list) -> str:
    convex_url = os.getenv("CONVEX_SITE_URL")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{convex_url}/addItemsToSprint",
                json={"sprintId": sprint_id, "taskIds": task_ids},
                timeout=10,
            )
            response.raise_for_status()
            return f"✅ Added {len(task_ids)} task(s) to sprint."
    except httpx.HTTPError as e:
        return f"❌ Failed to add tasks to sprint: {e}"


async def write_scheduler_to_convex(payload: dict) -> str:
    """Calls createOrUpdateScheduler Convex HTTP action."""
    convex_url = os.getenv("CONVEX_SITE_URL")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{convex_url}/createOrUpdateScheduler",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            return (
                f"✅ Scheduler saved — "
                f"name='{payload['name']}' "
                f"frequency={payload['frequencyDays']} days "
                f"recipientEmail='{payload.get('recipientEmail', 'owner email')}' "
                f"(id: {result.get('id', 'unknown')})"
            )
    except httpx.HTTPError as e:
        return f"❌ Failed to save scheduler: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# LLM MODELS
# ─────────────────────────────────────────────────────────────────────────────

_kaya_tools = [
    create_calendar_event,
    ask_project_analyst,
    create_sprint,
    add_items_to_sprint,
    setup_report_scheduler,
    get_scheduler,
    get_user_standup,
]

_kaya_llm_fast = ChatOpenAI(
    model=os.getenv("KAYA_FAST_MODEL", "gpt-4.1-mini"),
    temperature=0.4,
    streaming=True,
).bind_tools(_kaya_tools)

_kaya_llm_deep = ChatOpenAI(
    model=os.getenv("KAYA_DEEP_MODEL", "gpt-5.4-mini"),
    temperature=0.4,
    streaming=True,
).bind_tools(_kaya_tools)

# Analyst LLM — bound to its own read tools only, zero temperature for factual accuracy
_analyst_llm = ChatOpenAI(
    model=os.getenv("ANALYST_MODEL", "gpt-4.1-mini"),
    temperature=0,
    streaming=True,
).bind_tools(
    [
        get_tasks_summary,
        get_issues_summary,
        get_member_workload,
        get_sprint_insights,
        get_project_insights,
    ]
)

_mem0 = MemoryClient()


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────


KAYA_SYSTEM = """You are Kaya, an very Intelligent AI Product Manager and  a specialist in data interpretation.
what you can do -
1. you help teams with managing their project and ideas.
2. you look into their tasks / issues / sprints / Team member workloads and gather info and inform.
3. you understand their needs and create calendar events/ sprints / schedules for them.
4. you can set up automated report schedulers for projects.
 
Be concise, opinionated, and practical. Ask clarifying questions when needed.
- Always greet or refer to the user by their name if provided in the context.
- Be data-driven: use the counts (total, completed, blocked) to give an executive-level overview.
- Always try to repond in clear markdown points where it is necessary and useful.
 
 Here are your External Tools and Other subAgents for your help. usew them whenever you need them.
 
── Answering questions about tasks / issues / project status or team members workloads ──
Delegate to ask_project_analyst with a clear query and the active project_id.
Synthesise the findings into a helpful PM-level response.

── Preparing daily Standup ──
Use get_user_standup tool to get the user's daily standup whenever he asks his standup or what he needs to do today.
Follow the instructions from the tool output to present the standup to the user.
 
── Creating calendar events ──
Use create_calendar_event directly. No confirmation needed.
event_type must be 'event' or 'milestone'.
Dont ask for ISO dates and Time , all event by default are full day event if user dont specifically mentions time with dates.
Use ISO 8601 dates (e.g. 2025-04-22T00:00:00) but dont ask from user.Ask only simple date that users can comfotablly give.
If a tool call fails, retry with corrected parameters.
 
── Creating a sprint — follow this EXACT sequence, no skipping ──
Step 1: Call ask_project_analyst with query="analyze project and sprint status for planning" and project_id.
        This returns: project timeline, deadlines, and previous sprint performance.
Step 2: Tell the user what you found in very Imformative and Intelligent way. Example:
        "Your project deadline is in 14 days. There are 8 tasks ready to sprint.
         Last sprint was 'sprint-ui'. Tell me: sprint name, goal, start date and end date (max end date: <deadline>)."
Step 3: Wait for the user to reply with sprint name, goal, start date, end date.
Step 4: Call create_sprint with those exact values. Do NOT call it before the user replies.
Step 5: Once create_sprint returns a sprint_id, immediately call add_items_to_sprint(sprint_id).
        Do NOT ask the user for task IDs — the UI handles task selection automatically.
Step 6: After add_items_to_sprint completes, confirm to the user with the sprint name and task count.

── Setting up a report scheduler ──
Step 1: First always check the scheduler details by calling get_scheduler(project_id).
Step 2: Then tell user about the current scheduler details and call setup_report_scheduler(project_id).
Step 3: Once setup_report_scheduler returns, confirm the new configuration back to the user.
Step 4: In your final response, explicitly mention the frequency and the recipient email (e.g., "runs every 5 days and sends reports to team@example.com"). If they didn't specify an email, mention it goes to the project owner by default. Inform them they can always adjust these settings via the form."""


_ANALYST_SYSTEM = """You are the Project Analyst — a specialist in data interpretation.

Available tools:
- get_tasks_summary: Use for general task health, bottlenecks, and "what's stuck."
- get_issues_summary: Use for bug tracking, critical blockers, and stability.
- get_member_workload: Use to see who is doing what and if anyone is overloaded.
- get_sprint_insights: Use to analyze sprint velocity, progress, and historical performance.
- get_project_insights: Use for project-wide timeline, deadlines, and tracking.

Rules:
1. Always pass project_id.
2. For any "status" or "how are we doing" query, start by getting both tasks and issues summaries.
3. Use project_insights to answer questions about deadlines or remaining time.
4. Use sprint_insights to review how previous or current sprints are performing.
5. Be data-driven: use the counts and stats to provide an executive-level overview.
6. Return concise structured answers: bullet points, tables, or counts."""


# ─────────────────────────────────────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────────────────────────────────────


def kaya(state: KayaState, config: RunnableConfig) -> dict:
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
    user_id_context = (
        f"\nYour user_id: {user_id}\nPass this when calling get_user_standup."
    )
    user_context = (
        f"\n\nYou are talking to {state.get('user_name', 'a user')}."
        if state.get("user_name")
        else ""
    )
    current_date = datetime.now().strftime("%B %d, %Y")
    date_context = f"\n\nToday is {current_date}."

    full_messages = [
        SystemMessage(
            content=KAYA_SYSTEM
            + memory_block
            + project_context
            + user_id_context
            + user_context
            + date_context
        )
    ] + messages

    model_type = config["configurable"].get("model", "fast")
    llm = _kaya_llm_deep if model_type == "deep" else _kaya_llm_fast

    print(f"[kaya] invoking {model_type} model")
    response = llm.invoke(full_messages)

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
    """Seeds the isolated analyst thread. Resets on every new invocation."""
    args = tool_call["args"]
    return {
        "_analyst_tool_call_id": tool_call["id"],
        "_analyst_messages": [
            _RESET,
            {"role": "system", "content": _ANALYST_SYSTEM},
            {
                "role": "user",
                "content": f"Project ID: {args.get('project_id', '')}\nQuestion: {args.get('query', '')}",
            },
        ],
    }


def analyst_think(state: KayaState) -> dict:
    """Analyst LLM turn — decides next tool or produces final answer."""
    analyst_messages = state.get("_analyst_messages", [])
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
    """Executes one read tool and appends result to analyst thread."""
    name = tool_call["name"]
    args = tool_call["args"]

    # ── Emit a custom event so the frontend can show this tool card ──────────
    # metadata.writes collapses parallel same-name nodes to one key, so we
    # use the custom event channel (which is append-safe) instead.
    try:
        from langgraph.config import get_stream_writer
        write = get_stream_writer()
        write({"analyst_tool_running": name})
    except Exception:
        pass  # Non-critical — UI just won't show this tool card

    if name == "get_tasks_summary":
        result = get_tasks_summary.invoke(args)
    elif name == "get_issues_summary":
        result = get_issues_summary.invoke(args)
    elif name == "get_member_workload":
        result = get_member_workload.invoke(args)
    elif name == "get_sprint_insights":
        result = get_sprint_insights.invoke(args)
    elif name == "get_project_insights":
        result = get_project_insights.invoke(args)
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
    """Injects analyst's final answer into Kaya's message thread."""
    return {"messages": [closing_msg]}


# -----------------------------------TOOLS----------------------------------
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


# ── Sprint create — simple write, no interrupt ────────────────────────────────
async def sprint_create(tool_call: dict) -> dict:
    """Creates the sprint directly. Kaya has all info from the user already."""
    args = tool_call["args"]

    # Convert YYYY-MM-DD dates to unix ms
    start_ms = int(datetime.strptime(args["start_date"], "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.strptime(args["end_date"], "%Y-%m-%d").timestamp() * 1000)

    print(
        f"[sprint_create] Creating '{args['sprint_name']}' {args['start_date']} → {args['end_date']}"
    )

    try:
        result = await write_sprint_to_convex(
            {
                "projectId": args["project_id"],
                "sprintName": args["sprint_name"],
                "sprintGoal": args["sprint_goal"],
                "startDate": start_ms,
                "endDate": end_ms,
            }
        )
        # Convex insertSprint returns the sprint _id directly
        sprint_id = result.get("sprint")
        print(f"[sprint_create] ✓ sprint_id={sprint_id}")

        return {
            "messages": [
                ToolMessage(
                    content=f"✅ Sprint '{args['sprint_name']}' created. sprint_id={sprint_id}",
                    tool_call_id=tool_call["id"],
                    name="create_sprint",
                )
            ]
        }
    except httpx.HTTPError as e:
        return {
            "messages": [
                ToolMessage(
                    content=f"❌ Failed to create sprint: {e}",
                    tool_call_id=tool_call["id"],
                    name="create_sprint",
                )
            ]
        }


# ── Sprint add items — interrupt waits for UI task selection ─────────────────
async def sprint_add_items(tool_call: dict) -> dict:
    """
    Fires interrupt() so the UI shows the task selection box.
    Handles the Convex write internally — Kaya never sees raw task IDs.
    Resume payload: { task_ids: ["id1", "id2", ...] }
    """
    args = tool_call["args"]
    sprint_id = args["sprint_id"]

    print(f"[sprint_add_items] Interrupting for task selection — sprint_id={sprint_id}")

    selection = interrupt(
        {
            "tool": "add_items_to_sprint",
            "sprint_id": sprint_id,
            "message": "Select the tasks you want to add to this sprint.",
        }
    )

    task_ids = selection.get("task_ids", []) if isinstance(selection, dict) else []
    print(f"[sprint_add_items] Resumed — tasks={len(task_ids)}")

    result_msg = await write_items_to_sprint(sprint_id, task_ids)

    return {
        "messages": [
            ToolMessage(
                content=result_msg,
                tool_call_id=tool_call["id"],
                name="add_items_to_sprint",
            )
        ]
    }


# ------------------KAYA READ TOOLS---------------------------------
async def kaya_read_tools(tool_call: dict) -> dict:
    """Executes simple read tools directly for Kaya — no subagent needed."""
    name = tool_call["name"]
    args = tool_call["args"]

    if name == "get_scheduler":
        result = get_scheduler.invoke(args)
    elif name == "get_user_standup":
        result = get_user_standup.invoke(args)
    else:
        result = {"error": f"Unknown read tool: {name}"}

    return {
        "messages": [
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=name,
            )
        ]
    }


# New node — fires interrupt so UI shows the form, then writes on resume
async def scheduler_setup(tool_call: dict) -> dict:
    args = tool_call["args"]
    project_id = args["project_id"]

    # Check for existing scheduler first to pre-fill the form
    existing_data = None
    try:
        result = get_scheduler.invoke({"project_id": project_id})
        if result.get("exists"):
            existing_data = {
                "name": result.get("name"),
                "frequencyDays": result.get("frequencyDays"),
                "recipientEmail": result.get("recipientEmail"),
                "isActive": result.get("isActive"),
            }
    except Exception as e:
        print(f"[scheduler_setup] Could not fetch existing scheduler: {e}")

    # Interrupt — UI shows the scheduler form
    form_data = interrupt(
        {
            "tool": "setup_report_scheduler",
            "project_id": project_id,
            "existing_data": existing_data,
            "message": "Configure your automated report scheduler.",
        }
    )

    if not isinstance(form_data, dict) or not form_data.get("name"):
        return {
            "messages": [
                ToolMessage(
                    content="❌ Scheduler setup cancelled.",
                    tool_call_id=tool_call["id"],
                    name="setup_report_scheduler",
                )
            ]
        }

    # recipientEmail is optional from frontend; if not provided, backend uses owner email
    payload = {
        "projectId": project_id,
        "name": form_data["name"],
        "frequencyDays": form_data["frequencyDays"],
        "isActive": form_data.get("isActive", True),
    }
    if form_data.get("recipientEmail"):
        payload["recipientEmail"] = form_data["recipientEmail"]

    result_msg = await write_scheduler_to_convex(payload)

    return {
        "messages": [
            ToolMessage(
                content=result_msg,
                tool_call_id=tool_call["id"],
                name="setup_report_scheduler",
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
            case "create_sprint":
                sends.append(Send("sprint_create", tc))
            case "add_items_to_sprint":
                sends.append(Send("sprint_add_items", tc))
            case "get_scheduler":
                sends.append(Send("kaya_read_tools", tc))
            case "setup_report_scheduler":
                sends.append(Send("scheduler_setup", tc))
            case "get_user_standup":
                sends.append(Send("kaya_read_tools", tc))

    return sends if sends else END


def analyst_route(state: KayaState):
    """Routes after analyst_think: more tool calls → analyst_tools, done → kaya."""
    analyst_messages = state.get("_analyst_messages", [])
    last = analyst_messages[-1] if analyst_messages else None

    if last and last.get("tool_calls"):
        return [Send("analyst_tools", tc) for tc in last["tool_calls"]]

    analyst_tool_call_id = state.get("_analyst_tool_call_id", "unknown")
    final_content = last.get("content", "No findings.") if last else "No findings."

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

    # Core
    g.add_node("kaya", kaya)
    g.add_node("tools", tools)  # calendar HITL

    # Project analyst (flattened ReAct — reads only)
    g.add_node("project_analyst", project_analyst)  # seeds analyst thread
    g.add_node("analyst_think", analyst_think)  # LLM decides next step
    g.add_node("analyst_tools", analyst_tools)  # executes one read tool
    g.add_node("analyst_done", analyst_done)  # closes loop → kaya
    g.add_node("kaya_read_tools", kaya_read_tools)
    g.add_node("scheduler_setup", scheduler_setup)

    # Sprint write nodes (Kaya owns all writes directly)
    g.add_node("sprint_create", sprint_create)  # simple write, no interrupt
    g.add_node(
        "sprint_add_items", sprint_add_items
    )  # write + interrupt for task selection

    # Kaya flow
    g.add_edge(START, "kaya")
    g.add_conditional_edges("kaya", assign_tool)
    g.add_edge("tools", "kaya")
    g.add_edge("sprint_create", "kaya")
    g.add_edge("sprint_add_items", "kaya")
    g.add_edge("kaya_read_tools", "kaya")
    g.add_edge("scheduler_setup", "kaya")

    # Analyst flow
    g.add_edge("project_analyst", "analyst_think")
    g.add_conditional_edges("analyst_think", analyst_route)
    g.add_edge("analyst_tools", "analyst_think")
    g.add_edge("analyst_done", "kaya")

    return g.compile(checkpointer=checkpointer)


graph = build_graph()
print("[graph] Kaya agent graph compiled successfully")
