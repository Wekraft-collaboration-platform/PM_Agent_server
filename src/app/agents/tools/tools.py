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
    print(f"[CONVEX TOOL] Agent requested for tool '{endpoint}' with payload: {payload}")
    convex_url = _get_convex_url()
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{convex_url}/{endpoint.lstrip('/')}", json=payload, timeout=12)
            r.raise_for_status()
            data = r.json()
            print(f"[CONVEX TOOL] [OK] Completed '{endpoint}' successfully.")
            return data
    except Exception as e:
        safe_e = str(e).encode("ascii", "replace").decode("ascii")
        print(f"[CONVEX TOOL] [FAIL] Error requesting '{endpoint}': {safe_e}")
        raise


def convex_post_sync(endpoint: str, payload: dict) -> dict:
    """Centralized sync HTTP caller for Convex actions/queries."""
    print(f"[CONVEX TOOL] Agent requested for tool '{endpoint}' with payload: {payload}")
    convex_url = _get_convex_url()
    try:
        with httpx.Client() as client:
            r = client.post(f"{convex_url}/{endpoint.lstrip('/')}", json=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
            print(f"[CONVEX TOOL] [OK] Completed '{endpoint}' successfully.")
            return data
    except Exception as e:
        safe_e = str(e).encode("ascii", "replace").decode("ascii")
        print(f"[CONVEX TOOL] [FAIL] Error requesting '{endpoint}': {safe_e}")
        raise




# ─────────────────────────────────────────────────────────────────────────────
# READ & ANALYTICS TOOLS (@tool)
# ─────────────────────────────────────────────────────────────────────────────

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
def get_tasks_summary(project_id: str) -> dict:
    """Fetch a slim summary of project tasks. Returns:
    - tasks[]: list of { title, assignee, deadlineStatus, priority }
    - totalCount, completedCount, blockedCount
    """
    print(f"[get_tasks_summary] querying — project={project_id}")
    try:
        data = convex_post_sync("getTasksSummary", {"projectId": project_id})
        print(f"[get_tasks_summary] ✓ returned {len(data.get('tasks', []))} tasks")
        return data
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
    """Fetch the active automated report scheduler configuration for a project."""
    print(f"[get_scheduler] querying — project={project_id}")
    try:
        data = convex_post_sync("getScheduler", {"projectId": project_id})
        scheduler = data.get("scheduler")
        return {"exists": False} if not scheduler else {"exists": True, **scheduler}
    except Exception as e:
        print(f"[get_scheduler] ✗ ERROR: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC PARALLEL FETCHERS (Used by Sub-Agents for concurrent execution)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_user_standup_async(project_id: str, user_id: str) -> dict:
    try:
        data = await convex_post_async("getUserStandup", {"projectId": project_id, "userId": user_id})
        return data.get("standup", {}) or {"status": "empty", "message": "No active tasks or standup items found."}
    except Exception as e:
        return {"error": f"Standup fetch error: {e}"}


async def fetch_tasks_summary_async(project_id: str) -> dict:
    try:
        data = await convex_post_async("getTasksSummary", {"projectId": project_id})
        return data if data else {"status": "empty", "message": "No tasks found."}
    except Exception as e:
        return {"error": f"Tasks summary fetch error: {e}"}


async def fetch_issues_summary_async(project_id: str) -> dict:
    try:
        data = await convex_post_async("getIssuesSummary", {"projectId": project_id})
        return data.get("issuesSummary", {}) or {"status": "empty", "message": "No active issues found."}
    except Exception as e:
        return {"error": f"Issues summary fetch error: {e}"}


async def fetch_member_workload_async(project_id: str) -> dict:
    try:
        data = await convex_post_async("getMemberWorkloadPYAgent", {"projectId": project_id})
        return {"members": data.get("members", [])}
    except Exception as e:
        return {"error": f"Member workload fetch error: {e}"}


async def fetch_sprint_insights_async(project_id: str) -> dict:
    try:
        data = await convex_post_async("getSprintInsights", {"projectId": project_id})
        return {"sprints": data.get("sprints", [])}
    except Exception as e:
        return {"error": f"Sprint insights fetch error: {e}"}


async def fetch_project_insights_async(project_id: str) -> dict:
    try:
        data = await convex_post_async("getProjectInsights", {"projectId": project_id})
        return data.get("projectInsights", {}) or {"status": "empty", "message": "No project insights available."}
    except Exception as e:
        return {"error": f"Project insights fetch error: {e}"}


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
    """Create a calendar event for a project."""
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
def bulk_create_tasks(project_id: str) -> str:
    """Bulk create tasks extracted from uploaded PRDs / documents."""
    return "intercepted"


@tool
def bulk_create_issues(project_id: str) -> str:
    """Bulk create issues extracted from uploaded documents or error logs."""
    return "intercepted"


# ─────────────────────────────────────────────────────────────────────────────
# CONVEX WRITE HELPERS (Called after HITL Approval)
# ─────────────────────────────────────────────────────────────────────────────

async def write_calendar_event_to_convex(payload: dict) -> str:
    """Actual HTTP call to Convex for calendar creation after HITL approval."""
    try:
        result = await convex_post_async("createCalendarEvent", payload)
        return f"✅ Calendar event created: '{payload.get('title')}' (id: {result.get('id', 'unknown')})"
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


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTED TOOLSETS (Categorized by Pure Concern)
# ─────────────────────────────────────────────────────────────────────────────

# All Mutation / Write / HITL Tools
DBWRITE_TOOLS = [
    create_calendar_event,
    bulk_create_tasks,
    bulk_create_issues,
]

# All Read-Only Analytics & Insights Tools
ANALYST_TOOLS = [
    get_user_standup,
    get_tasks_summary,
    get_issues_summary,
    get_member_workload,
    get_project_insights,
]

# Sprint Management Tools
SPRINT_TOOLS = [
    get_sprint_insights,
    create_sprint,
    add_items_to_sprint,
]

ALL_TOOLS = DBWRITE_TOOLS + ANALYST_TOOLS + SPRINT_TOOLS


# ─────────────────────────────────────────────────────────────────────────────
# SKILLS FETCHING & LEVEL 1 METADATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SKILLS_FALLBACK = [
    {
        "name": "sentry_error_triage",
        "title": "Sentry Error Triage & Root Cause Analysis",
        "connectorId": "sentry",
        "description": "Discovers Sentry organizations, fetches unresolved production errors with bounded queries, identifies culprit routes, and isolates critical system crashes.",
        "content": """---
name: sentry_error_triage
title: Sentry Error Triage & Root Cause Analysis
description: Discovers Sentry organizations, fetches unresolved production errors with bounded queries, identifies culprit routes, and isolates critical system crashes.
connectorId: sentry
createdBy: default
version: 1.0.0
---

# Sentry Error Triage Skill

## Context & Objectives
This skill governs how Kaya and MCP sub-agents query live Sentry instances to diagnose unresolved exceptions, trace offending routes, measure affected user blast-radius, and prioritize production incidents.

## Discovery & Prerequisites Protocol
1. Never guess the Organization Slug: Always run find_organizations first to discover the real active slug.
2. Verify Target Project: Execute find_projects with the discovered organizationSlug. Match with the active project or target repository.

## Step-by-Step Tool Execution Workflow
1. Step 1 - Discover Organization: Tool: find_organizations
2. Step 2 - Discover Project Slugs: Tool: find_projects
3. Step 3 - Query Unresolved Issues (Bounded): Tool: search_issues with {"query": "is:unresolved", "sort": "date"}
4. Step 4 - Deep Dive on Outlier Issues (Optional): Tool: get_sentry_resource or analyze_issue_with_seer

## Strict Anti-Hallucination & Execution Rules
- No Fabricated Error Counts: You must only report exact events and users numbers returned by Sentry.
- Link Preservation: Always extract and provide the direct Sentry dashboard permalink.
- Finalize execution with finalize_result containing data, resolved_context, and summary.
""",
    },
    {
        "name": "linear_issue_management",
        "title": "Linear Issue Sync & Safe Ticket Creation",
        "connectorId": "linear",
        "description": "Discovers Linear workspace teams, lists active/backlog tickets without truncation, safely creates tickets with required team UUIDs, and verifies creation.",
        "content": """---
name: linear_issue_management
title: Linear Issue Sync & Safe Ticket Creation
description: Discovers Linear workspace teams, lists active/backlog tickets without truncation, safely creates tickets with required team UUIDs, and verifies creation.
connectorId: linear
createdBy: default
version: 1.0.0
---

# Linear Issue Management Skill

## Context & Objectives
This skill instructs Kaya and MCP sub-agents on how to interact with Linear: discovering workspace teams, synchronizing active/backlog issues, safely creating new tickets with required team identifiers, and preventing duplicate tickets.

## Discovery & Prerequisites Protocol
1. Team ID is Mandatory for Mutations: Linear requires a valid team UUID to create any issue. Always invoke list_teams if teamId is unknown.
2. Never guess IDs: Never pass an arbitrary string as team without verifying it via list_teams.

## Step-by-Step Tool Execution Workflow
1. Step 1 - Team Discovery: Tool: list_teams
2. Step 2 - Sync Active Issues: Tool: list_issues (filter status != completed/canceled)
3. Step 3 - Safe Issue Creation: Tool: save_issue with discovered team ID
4. Step 4 - Read-After-Write Verification: Tool: list_issues to verify created issue key

## Strict Anti-Hallucination & Execution Rules
- No Speculative Creation: Never claim an issue was created until save_issue returns a valid payload.
- Deduplication Check: Cross-reference existing ticket titles before creating.
- Finalize execution with finalize_result containing data, resolved_context, and summary.
""",
    },
    {
        "name": "incident_escalation_triage",
        "title": "Cross-App Production Incident Escalation",
        "connectorId": "sentry, linear, project",
        "description": "Coordinates Sentry telemetry, Linear ticketing, and internal project issues to escalate critical production crashes (>10 events) into actionable engineering tickets.",
        "content": """---
name: incident_escalation_triage
title: Cross-App Production Incident Escalation
description: Coordinates Sentry telemetry, Linear ticketing, and internal project issues to escalate critical production crashes (>10 events) into actionable engineering tickets.
connectorId: sentry, linear, project
createdBy: default
version: 1.0.0
---

# Cross-App Incident Escalation Skill

## Context & Objectives
Coordinates Sentry telemetry, Linear ticketing, and internal project database to detect critical production anomalies and escalate them to the backlog.

## Incident Escalation Thresholds
An error is Escalation-Ready when:
1. events >= 10 OR users >= 3 in Sentry.
2. OR error is an unhandled fatal crash.
3. AND no open issue currently exists in Linear or internal project database.

## Step-by-Step Multi-System Workflow
1. Phase 1 - Telemetry Ingestion: Query Sentry for active unresolved crashes.
2. Phase 2 - Deduplication & Existing Issue Scan: Check open Linear tickets and internal project issues.
3. Phase 3 - Ticket Escalation: If untracked, invoke save_issue via Linear MCP or stage internal project issue.
4. Phase 4 - Verification & Audit Log: Verify creation via list_issues.
- Finalize execution with finalize_result containing incident summary and escalation receipts.
""",
    },
    {
        "name": "cross_platform_issue_triage_notion_sync",
        "title": "Linear & Sentry Issue Triage & Notion Sync",
        "connectorId": "linear, sentry, notion",
        "description": "Fetches active issues from Linear and unresolved production errors from Sentry, aggregates them into structured markdown tables, and creates a Notion page.",
        "content": """---
name: cross_platform_issue_triage_notion_sync
title: Linear & Sentry Issue Triage & Notion Sync
connectorId: linear, sentry, notion
version: 1.3.0
---

# Linear & Sentry Issue Triage & Notion Sync

## Execution Steps
1. Linear -> call list_issues with {}. Extract id, title, priority, state, url.
2. Sentry -> call find_organizations -> capture organizationSlug -> call search_issues with {organizationSlug, query: "is:unresolved", sort: "date"}. Extract error title, culprit, event count, url.
3. Notion -> call notion-create-pages with {creation_mode: "draft", allow_async: false, pages: [{properties: {title: "Linear & Sentry Issue Triage - Live Sync"}, content: "<markdown>", icon: "📊"}]}. Capture returned id and url.
4. Finalize -> call finalize_result with the created Notion page URL and issue counts.

## Rules
- Only output IDs, URLs, and counts that appear verbatim in tool responses. Never invent keys or links.
- End with finalize_result containing the Notion page URL, issue counts, and top-priority highlights.

## Execution Configuration
```json
{
  "maxSteps": 2,
  "pinnedTools": {
    "linear": [
      {
        "name": "list_issues",
        "description": "List all Linear issues. Returns issues array with id, title, priority, state, url.",
        "inputSchema": { "type": "object", "properties": {}, "required": [] }
      }
    ],
    "sentry": [
      {
        "name": "find_organizations",
        "description": "Find Sentry orgs. Returns organizations array each with a slug. Call first before search_issues.",
        "inputSchema": { "type": "object", "properties": {}, "required": [] }
      },
      {
        "name": "search_issues",
        "description": "Search Sentry issues. Requires organizationSlug. Use query 'is:unresolved', sort 'date'.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "organizationSlug": { "type": "string" },
            "query": { "type": "string" },
            "sort": { "type": "string" }
          },
          "required": ["organizationSlug", "query"]
        }
      }
    ],
    "notion": [
      {
        "name": "notion-create-pages",
        "description": "Create Notion pages. Pass creation_mode 'draft', allow_async false for sync id return.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "pages": { "type": "array", "items": { "type": "object" } },
            "creation_mode": { "type": "string", "enum": ["draft"] },
            "allow_async": { "type": "boolean" }
          },
          "required": ["pages"]
        }
      }
    ]
  }
}
```""",
    },
    {
        "name": "jira_project_tasks_alignment_notion_sync",
        "title": "Project & Jira Tasks Alignment & Notion Action Plan",
        "connectorId": "jira, notion, project",
        "description": "Fetches internal project tasks and live Jira tasks via JQL, performs cross-system alignment and workload gap analysis, generates actionable insights and recommended actions, and publishes a structured report to Notion.",
        "content": """---
name: jira_project_tasks_alignment_notion_sync
title: Project & Jira Tasks Alignment & Notion Action Plan
connectorId: jira, notion, project
version: 1.0.0
---

# Project & Jira Tasks Alignment & Notion Action Plan

## Context & Objectives
This skill coordinates internal project tasks and external Jira Kanban/Scrum tasks to identify unassigned items, status mismatches, and delivery blockers, synthesizing them into actionable insights and publishing a strategic plan to Notion.

## Step-by-Step Execution Workflow
1. Jira Discovery: Call getAccessibleAtlassianResources with {} to resolve cloudId.
2. Jira Tasks Query: Call searchJiraIssuesUsingJql with:
   Default JQL: { "cloudId": "<discovered_cloudId>", "jql": "status IS NOT NULL ORDER BY updated DESC", "maxResults": 20 }
   (Do not filter by project='...' by default so it works for any workspace without project key mismatch).
   If User Specified Project: Use jql "project = '<KeyOrName>' AND status IS NOT NULL ORDER BY updated DESC". If error, fall back to "status IS NOT NULL ORDER BY updated DESC".
   Important: Search with status IS NOT NULL to retrieve all active tasks (including unassigned items). Extract key, summary, status.name, priority.name, assignee.displayName.
3. Internal Project Tasks: Ingest active project tasks, priorities, assignees, and deadlines.
4. Strategic Synthesis: Formulate 3-5 concrete Actionable Insights and 2-3 immediate Recommended Actions.
5. Notion Publication: Call notion-create-pages with draft creation_mode and the structured markdown content.
6. Finalization Handshake: Call finalize_result with created Notion page URL and strategic highlights.

## Execution Configuration
```json
{
  "maxSteps": 2,
  "pinnedTools": {
    "jira": [
      {
        "name": "getAccessibleAtlassianResources",
        "description": "Discover accessible Atlassian cloudId and resources. Call first before running JQL.",
        "inputSchema": { "type": "object", "properties": {}, "required": [] }
      },
      {
        "name": "searchJiraIssuesUsingJql",
        "description": "Search Jira issues using JQL. Pass cloudId and bounded jql 'status IS NOT NULL ORDER BY updated DESC'.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "cloudId": { "type": "string" },
            "jql": { "type": "string" },
            "maxResults": { "type": "number" }
          },
          "required": ["cloudId", "jql"]
        }
      }
    ],
    "notion": [
      {
        "name": "notion-create-pages",
        "description": "Create Notion pages. Pass creation_mode 'draft', allow_async false for sync id return.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "pages": { "type": "array", "items": { "type": "object" } },
            "creation_mode": { "type": "string", "enum": ["draft"] },
            "allow_async": { "type": "boolean" }
          },
          "required": ["pages"]
        }
      }
    ]
  }
}
```""",
    },
    {
        "name": "notion_page_operations",
        "title": "Notion — Create, Update & Fetch Pages",
        "connectorId": "notion",
        "description": "Exact MCP tool names, payloads, and response parsing to create a Notion page, insert content into it, and fetch its full content.",
        "content": """---
name: notion_page_operations
title: Notion — Create, Update & Fetch Pages
connectorId: notion
version: 1.0.0
---

# Notion Page Operations

> ⚠ All Notion MCP responses wrap data as a JSON string inside `result.content`. Always parse `result["content"]` as JSON before reading any field.

---

## Step 1 — Create a Page
Tool: `notion-create-pages`
```json
{
  "creation_mode": "draft",
  "allow_async": false,
  "pages": [{
    "properties": { "title": "<Page Title>" },
    "content": "<Markdown body>",
    "icon": "📋"
  }]
}
```
Extract ID from response:
```python
inner = json.loads(result["content"])
page_id = inner["pages"][0]["id"]
page_url = inner["pages"][0]["url"]
```

---

## Step 2 — Insert Content into a Page
Tool: `notion-update-page`
```json
{
  "page_id": "<pageId>",
  "command": "insert_content",
  "content": "<Markdown to append>"
}
```
> ⚠ Field is `page_id` — NOT `id`. Wrong key causes validation error.
> command options: `insert_content` | `replace_content` | `update_properties`

Success: `result["isError"] == false`

---

## Step 3 — Fetch Page Content
Tool: `notion-fetch`
```json
{ "id": "<pageId or full https://app.notion.com/p/<id> URL>" }
```
Read body:
```python
inner = json.loads(result["content"])
body = inner["text"]  # full page content
```

---

## Finalization
Call `finalize_result` with `{ pageId, pageUrl }` from Step 1 and a one-line summary.

## Rules
- Never claim page was created until `notion-create-pages` returns `isError: false`.
- Never fabricate URLs — always use the `url` from the create response.""",
    },
]



async def fetch_active_skills_async(user_id: str = "") -> List[Dict[str, Any]]:
    """
    Fetches active skills (Level 1 metadata + content) for the user from Convex HTTP backend.
    Falls back to platform default skills if backend is unreachable.
    """
    try:
        data = await convex_post_async("getActiveSkills", {"userId": user_id or None})
        skills = data.get("skills") if isinstance(data, dict) else data
        if isinstance(skills, list) and skills:
            print(f"[FETCH SKILLS] [OK] Retrieved {len(skills)} skill(s) from Convex DB.")
            return skills
    except Exception as e:
        safe_e = str(e).encode("ascii", "replace").decode("ascii")
        print(f"[FETCH SKILLS] Notice: Could not fetch skills from Convex endpoint ({safe_e}), using default skills catalog.")
    return DEFAULT_SKILLS_FALLBACK

