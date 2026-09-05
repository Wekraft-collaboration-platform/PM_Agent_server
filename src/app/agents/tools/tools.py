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
    """
    print(f"[get_scheduler] querying — project={project_id}")
    try:
        data = convex_post_sync("getScheduler", {"projectId": project_id})
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
    except Exception as e:
        print(f"[get_scheduler] ✗ ERROR: {e}")
        return {"exists": False, "error": str(e)}


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
    All events are created as all-day events by default.
    Never ask the user for project_id — it is injected automatically.
    """
    return "intercepted"


@tool
def ask_project_analyst(query: str, project_id: str) -> str:
    """Delegate a data question to the Project Analyst subagent.

    The analyst has read access to the project's tasks and issues and can answer questions like:
    - "What tasks are blocked?"
    - "Show me all critical issues in production"
    - "Which high-priority tasks are still not started?"
    - "Which team member has the most open issues?"
    - "Summarise sprint progress"
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
    2. The user has provided sprint_name, sprint_goal, start_date and end_date (YYYY-MM-DD).
    """
    return "intercepted"


@tool
def add_items_to_sprint(sprint_id: str) -> str:
    """Trigger the item selection UI so the user can pick tasks for the sprint.

    Call immediately after create_sprint succeeds.
    The UI shows all available tasks — user selects and confirms.
    """
    return "intercepted"


@tool
def setup_report_scheduler(project_id: str) -> str:
    """Open the scheduler setup form for the user to configure automated reports.

    Call this when the user wants to:
    - Set up automated / scheduled reports for a project
    - Change how often reports are generated
    - Enable or disable an existing scheduler
    - Set or update the recipient email for reports
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
            f"recipientEmail='{payload.get('recipientEmail', 'owner email')}' "
            f"(id: {result.get('id', 'unknown')})"
        )
    except Exception as e:
        return f"❌ Failed to save scheduler: {e}"


# Exported toolsets
ALL_TOOLS = [
    get_tasks_summary,
    get_issues_summary,
    get_member_workload,
    get_user_standup,
    get_sprint_insights,
    get_project_insights,
    get_scheduler,
    create_calendar_event,
    ask_project_analyst,
    create_sprint,
    add_items_to_sprint,
    setup_report_scheduler,
]
