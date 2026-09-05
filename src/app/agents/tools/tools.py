import os
import httpx
from typing import Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Load Convex Site HTTP URL from environment
CONVEX_SITE_URL: str = os.getenv("CONVEX_SITE_URL", "http://127.0.0.1:3000")


def _get_convex_url() -> str:
    return os.getenv("CONVEX_SITE_URL", CONVEX_SITE_URL).rstrip("/")


# ─────────────────────────────────────────────────────────────────────────────
# CONVEX HTTP RETRY HELPER
# ─────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
async def convex_post_async(endpoint: str, payload: dict) -> dict:
    """Centralized async HTTP caller for Convex actions/queries with automatic retry logic."""
    convex_url = _get_convex_url()
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{convex_url}/{endpoint.lstrip('/')}", json=payload, timeout=12)
        r.raise_for_status()
        return r.json()


def convex_post_sync(endpoint: str, payload: dict) -> dict:
    """Centralized sync HTTP caller for Convex actions/queries."""
    convex_url = _get_convex_url()
    with httpx.Client() as client:
        r = client.post(f"{convex_url}/{endpoint.lstrip('/')}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY TOOLS (@tool)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_user_memory(user_id: str, query: str) -> dict:
    """Search long-term memory for user preferences, past choices, and user context.
    Use this whenever the user asks about their personal preferences, past interactions, or stored facts.
    """
    print(f"[search_user_memory] Searching Mem0 for user_id={user_id}, query='{query}'")
    try:
        mem0_key = os.getenv("MEM0_API_KEY")
        if not mem0_key:
            return {"memories": [], "note": "MEM0_API_KEY not set"}
        from mem0 import MemoryClient
        client = MemoryClient(api_key=mem0_key)
        results = client.search(query, user_id=user_id)
        print(f"[search_user_memory] ✓ Found {len(results)} memory entries")
        return {"memories": results}
    except Exception as e:
        print(f"[search_user_memory] ✗ Error searching Mem0: {e}")
        return {"memories": [], "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# READ TOOLS (@tool)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_tasks_summary(project_id: str) -> dict:
    """Fetch a high-level summary of all tasks including critical and active ones.
    Useful for getting a quick overview of project health and identifying bottlenecks.
    """
    print(f"[get_tasks_summary] querying — project={project_id}")
    try:
        data = convex_post_sync("getTasksSummary", {"projectId": project_id})
        summary = data.get("tasksSummary", {})
        print(f"[get_tasks_summary] ✓ returned")
        return summary
    except Exception as e:
        print(f"[get_tasks_summary] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_issues_summary(project_id: str) -> dict:
    """Fetch a summary of all issues, focusing on active and critical ones.
    Useful for identifying major blockers and critical bugs.
    """
    print(f"[get_issues_summary] querying — project={project_id}")
    try:
        data = convex_post_sync("getIssuesSummary", {"projectId": project_id})
        summary = data.get("issuesSummary", {})
        print(f"[get_issues_summary] ✓ returned")
        return summary
    except Exception as e:
        print(f"[get_issues_summary] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_member_workload(project_id: str) -> dict:
    """Returns a detailed breakdown of each team member's current task and issue assignments.
    Useful for load balancing and seeing who is busy.
    """
    print(f"[get_member_workload] querying — project={project_id}")
    try:
        data = convex_post_sync("getMemberWorkloadPYAgent", {"projectId": project_id})
        members = data.get("members", [])
        print(f"[get_member_workload] ✓ {len(members)} members returned")
        return {"members": members}
    except Exception as e:
        print(f"[get_member_workload] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_user_standup(project_id: str, user_id: str) -> dict:
    """Fetch active tasks and open issues assigned to a specific user.
    Useful for daily standups and helping the user prioritize their work for today and tomorrow.
    """
    print(f"[get_user_standup] querying — project={project_id} user={user_id}")
    try:
        data = convex_post_sync("getUserStandup", {"projectId": project_id, "userId": user_id})
        standup = data.get("standup", {})
        print(f"[get_user_standup] ✓ returned")
        return standup
    except Exception as e:
        print(f"[get_user_standup] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_sprint_insights(project_id: str) -> dict:
    """Fetch comprehensive analytics for all project sprints, including progress metrics and timelines.
    Useful for understanding sprint velocity and overall progress.
    """
    print(f"[get_sprint_insights] querying — project={project_id}")
    try:
        data = convex_post_sync("getSprintInsights", {"projectId": project_id})
        sprints = data.get("sprints", [])
        print(f"[get_sprint_insights] ✓ {len(sprints)} sprints returned")
        return {"sprints": sprints}
    except Exception as e:
        print(f"[get_sprint_insights] ✗ ERROR: {e}")
        return {"error": str(e)}


@tool
def get_project_insights(project_id: str) -> dict:
    """Fetch basic project timeline information like deadline and days remaining.
    Useful for overall project status and tracking against the final deadline.
    """
    print(f"[get_project_insights] querying — project={project_id}")
    try:
        data = convex_post_sync("getProjectInsights", {"projectId": project_id})
        insights = data.get("projectInsights", {})
        print(f"[get_project_insights] ✓ returned")
        return insights
    except Exception as e:
        print(f"[get_project_insights] ✗ ERROR: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# WRITE / HITL SCHEMA DESCRIPTOR TOOLS (@tool)
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
    """Create a calendar event for a project.
    event_type must be one of: event, milestone.
    start_iso and end_iso must be ISO 8601 strings.
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
    """Create a new sprint for the project."""
    return "intercepted"


@tool
def add_items_to_sprint(sprint_id: str) -> str:
    """Trigger the item selection UI so the user can pick tasks for the sprint."""
    return "intercepted"


@tool
def setup_report_scheduler(project_id: str) -> str:
    """Open or view the report scheduler form for a project."""
    return "intercepted"


@tool
def bulk_create_tasks(project_id: str) -> str:
    """Bulk create tasks extracted from uploaded PRDs / documents.
    Agent specifies 4 key fields per task: title, description, priority, type.
    Status defaults to 'not started' and estimation to today-tomorrow.
    """
    return "intercepted"


@tool
def bulk_create_issues(project_id: str) -> str:
    """Bulk create issues extracted from uploaded documents or error logs.
    Agent specifies 4 key fields per issue: title, description, environment, severity.
    Status defaults to 'not opened' and due_date to day after tomorrow.
    """
    return "intercepted"


# ─────────────────────────────────────────────────────────────────────────────
# CONVEX WRITE HELPERS (Called after HITL Approval)
# ─────────────────────────────────────────────────────────────────────────────

async def write_calendar_event_to_convex(payload: dict) -> str:
    """Actual HTTP call to Convex for calendar creation after HITL approval."""
    try:
        result = await convex_post_async("createCalendarEvent", payload)
        return (
            f"✅ Calendar event created: '{payload.get('title')}' "
            f"(id: {result.get('id', 'unknown')})"
        )
    except Exception as e:
        return f"❌ Failed to create calendar event: {e}"


async def write_sprint_to_convex(payload: dict) -> dict:
    """Actual HTTP call to Convex for sprint creation after HITL approval."""
    return await convex_post_async("createSprint", payload)


async def write_items_to_sprint(sprint_id: str, task_ids: list) -> str:
    """Actual HTTP call to Convex for adding sprint items after HITL approval."""
    try:
        await convex_post_async("addItemsToSprint", {"sprintId": sprint_id, "taskIds": task_ids})
        return f"✅ Added {len(task_ids)} task(s) to sprint."
    except Exception as e:
        return f"❌ Failed to add tasks to sprint: {e}"


async def write_scheduler_to_convex(payload: dict) -> str:
    """Actual HTTP call to Convex for scheduler configuration after HITL approval."""
    try:
        result = await convex_post_async("createOrUpdateScheduler", payload)
        return (
            f"✅ Scheduler saved — "
            f"name='{payload.get('name')}' "
            f"frequency={payload.get('frequencyDays')} days "
            f"(id: {result.get('id', 'unknown')})"
        )
    except Exception as e:
        return f"❌ Failed to save scheduler: {e}"


async def write_bulk_tasks_to_convex(payload: dict) -> str:
    """Actual HTTP call to Convex for bulk task insertion after HITL approval."""
    try:
        result = await convex_post_async("bulkInsertTasks", payload)
        return f"✅ Bulk created {result.get('count', 0)} task(s) in project."
    except Exception as e:
        return f"❌ Failed to bulk create tasks: {e}"


async def write_bulk_issues_to_convex(payload: dict) -> str:
    """Actual HTTP call to Convex for bulk issue insertion after HITL approval."""
    try:
        result = await convex_post_async("bulkInsertIssues", payload)
        return f"✅ Bulk created {result.get('count', 0)} issue(s) in project."
    except Exception as e:
        return f"❌ Failed to bulk create issues: {e}"


# Exported Agent Toolsets
KAYA_TOOLS = [
    search_user_memory,
    create_calendar_event,
    get_user_standup,
    setup_report_scheduler,
]

ANALYST_TOOLS = [
    get_tasks_summary,
    get_issues_summary,
    get_member_workload,
    get_project_insights,
    bulk_create_tasks,
    bulk_create_issues,
]

SPRINT_TOOLS = [
    get_sprint_insights,
    create_sprint,
    add_items_to_sprint,
]

ALL_TOOLS = KAYA_TOOLS + ANALYST_TOOLS + SPRINT_TOOLS
