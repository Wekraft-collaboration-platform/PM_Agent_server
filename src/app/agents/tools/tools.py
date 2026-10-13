import os
import httpx
import asyncio
from datetime import datetime, timedelta, timezone
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


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB ASYNC FETCHERS & TOOLS (Direct REST API with Clerk OAuth Token)
# ─────────────────────────────────────────────────────────────────────────────
# GITHUB TOOLS (PRs, Issues, Contributor Velocity, Release & CI Health)
# ─────────────────────────────────────────────────────────────────────────────

async def get_github_access_token_async(user_id: str = "") -> str:
    """Auto-resolves user GitHub OAuth access token directly from Clerk."""
    clerk_key = os.getenv("CLERK_SECRET_KEY", "").strip()
    if not clerk_key:
        try:
            from pathlib import Path
            client_env = Path(r"r:\exp_wekraft\client\.env.local")
            if client_env.exists():
                for line in client_env.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("CLERK_SECRET_KEY="):
                        clerk_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass

    if not clerk_key:
        return ""

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            headers = {"Authorization": f"Bearer {clerk_key}"}
            if user_id and user_id.startswith("user_"):
                tok_r = await client.get(
                    f"https://api.clerk.com/v1/users/{user_id}/oauth_access_tokens/oauth_github",
                    headers=headers,
                )
                if tok_r.status_code == 200:
                    toks = tok_r.json()
                    if toks and len(toks) > 0 and toks[0].get("token"):
                        return toks[0].get("token")

            r = await client.get("https://api.clerk.com/v1/users?limit=20", headers=headers)
            if r.status_code == 200:
                for u in r.json():
                    uid = u.get("id")
                    tok_r = await client.get(
                        f"https://api.clerk.com/v1/users/{uid}/oauth_access_tokens/oauth_github",
                        headers=headers,
                    )
                    if tok_r.status_code == 200:
                        toks = tok_r.json()
                        if toks and len(toks) > 0 and toks[0].get("token"):
                            return toks[0].get("token")
    except Exception as e:
        print(f"[GITHUB TOOL] Notice: Clerk token lookup: {e}")

    return ""


def _get_github_headers(token: str) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "WeKraft-Kaya-Agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def fetch_github_pull_requests_async(owner: str, repo: str, token: str = "") -> dict:
    """Fetch pull requests summary: open PRs, stale PRs (>48h), draft PRs, and merged PRs."""
    if not token:
        token = await get_github_access_token_async()

    if not owner or not repo:
        return {"error": "Missing repository owner or repo name."}
    if not token:
        return {"error": "GitHub is not linked or unauthorized. Please connect your GitHub account."}
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&sort=updated&direction=desc&per_page=30"
    headers = _get_github_headers(token)

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return {"error": f"GitHub API error ({resp.status_code}): {resp.text}"}
            pulls = resp.json()

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        open_prs = []
        stale_prs = []
        merged_prs = []

        for p in pulls:
            state = p.get("state", "open")
            is_draft = p.get("draft", False)
            created_str = p.get("created_at") or ""
            updated_str = p.get("updated_at") or ""
            merged_at_str = p.get("merged_at")

            is_stale = False
            if state == "open" and updated_str:
                try:
                    updated_dt = datetime.fromisoformat(updated_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    diff_hours = (now - updated_dt).total_seconds() / 3600.0
                    if diff_hours > 48.0:
                        is_stale = True
                except Exception:
                    pass

            pr_item = {
                "number": p.get("number"),
                "title": p.get("title"),
                "author": p.get("user", {}).get("login", "unknown"),
                "state": state,
                "draft": is_draft,
                "is_stale": is_stale,
                "created_at": created_str,
                "updated_at": updated_str,
                "head_branch": p.get("head", {}).get("ref", ""),
                "base_branch": p.get("base", {}).get("ref", ""),
                "html_url": p.get("html_url", ""),
                "requested_reviewers": [r.get("login") for r in p.get("requested_reviewers", []) if r.get("login")],
            }

            if state == "open":
                open_prs.append(pr_item)
                if is_stale:
                    stale_prs.append(pr_item)
            elif merged_at_str or state == "closed":
                merged_prs.append({
                    "number": p.get("number"),
                    "title": p.get("title"),
                    "author": p.get("user", {}).get("login", "unknown"),
                    "merged_at": merged_at_str or updated_str,
                    "html_url": p.get("html_url", ""),
                })

        return {
            "total_tracked": len(pulls),
            "open_count": len(open_prs),
            "stale_count": len(stale_prs),
            "merged_count": len(merged_prs),
            "open_prs": open_prs,
            "stale_prs": stale_prs,
            "recent_merged_prs": merged_prs[:5],
        }
    except Exception as e:
        return {"error": f"Failed to fetch pull requests: {e}"}


async def fetch_github_issues_async(owner: str, repo: str, token: str = "", state: str = "all") -> dict:
    """Fetch repository issues summary (excluding PRs), identifying unassigned and blocker bugs."""
    if not token:
        token = await get_github_access_token_async()

    if not owner or not repo:
        return {"error": "Missing repository owner or repo name."}
    if not token:
        return {"error": "GitHub is not linked or unauthorized. Please connect your GitHub account."}

    url = f"https://api.github.com/repos/{owner}/{repo}/issues?state={state}&sort=updated&direction=desc&per_page=30"
    headers = _get_github_headers(token)

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return {"error": f"GitHub API error ({resp.status_code}): {resp.text}"}
            raw_items = resp.json()

        issues = []
        open_issues = []
        closed_issues = []
        unassigned_issues = []
        blocker_issues = []

        for item in raw_items:
            # Filter out pull requests
            if item.get("pull_request"):
                continue

            labels = [lbl.get("name", "") if isinstance(lbl, dict) else str(lbl) for lbl in item.get("labels", [])]
            assignees = [a.get("login", "") for a in item.get("assignees", []) if a.get("login")]
            if not assignees and item.get("assignee"):
                assignees = [item["assignee"].get("login", "")]

            is_unassigned = len(assignees) == 0
            is_blocker = any(any(k in lbl.lower() for k in ["bug", "critical", "blocker", "urgent", "p0", "high"]) for lbl in labels)

            iss_data = {
                "number": item.get("number"),
                "title": item.get("title"),
                "state": item.get("state", "open"),
                "author": item.get("user", {}).get("login", "unknown"),
                "assignees": assignees,
                "is_unassigned": is_unassigned,
                "is_blocker": is_blocker,
                "labels": labels,
                "comments_count": item.get("comments", 0),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "html_url": item.get("html_url", ""),
            }

            issues.append(iss_data)
            if iss_data["state"] == "open":
                open_issues.append(iss_data)
                if is_unassigned:
                    unassigned_issues.append(iss_data)
                if is_blocker:
                    blocker_issues.append(iss_data)
            else:
                closed_issues.append(iss_data)

        return {
            "total_count": len(issues),
            "open_count": len(open_issues),
            "closed_count": len(closed_issues),
            "unassigned_count": len(unassigned_issues),
            "blocker_count": len(blocker_issues),
            "open_issues": open_issues,
            "unassigned_issues": unassigned_issues,
            "blocker_issues": blocker_issues,
        }
    except Exception as e:
        return {"error": f"Failed to fetch issues: {e}"}


async def fetch_github_contributor_activity_async(owner: str, repo: str, token: str = "", days: int = 14) -> dict:
    """Fetch team commit frequency, active contributors, and recent commits."""
    if not token:
        token = await get_github_access_token_async()

    if not owner or not repo:
        return {"error": "Missing repository owner or repo name."}
    if not token:
        return {"error": "GitHub is not linked or unauthorized. Please connect your GitHub account."}

    seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).replace(tzinfo=None)
    timeframe_ago = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)

    url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page=50"
    headers = _get_github_headers(token)

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return {"error": f"GitHub API error ({resp.status_code}): {resp.text}"}
            commits = resp.json()

        author_map: Dict[str, Dict[str, Any]] = {}
        commits_last_7_days = 0
        commits_in_timeframe = 0

        for c in commits:
            author_info = c.get("author") or {}
            commit_info = c.get("commit", {}).get("author", {}) or {}

            author_login = author_info.get("login") or commit_info.get("name") or "unknown"
            author_name = commit_info.get("name") or author_login
            avatar_url = author_info.get("avatar_url") or ""
            date_str = commit_info.get("date") or ""

            if date_str:
                try:
                    c_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")).replace(tzinfo=None)
                    if c_dt >= seven_days_ago:
                        commits_last_7_days += 1
                    if c_dt >= timeframe_ago:
                        commits_in_timeframe += 1
                except Exception:
                    pass

            if author_login not in author_map:
                author_map[author_login] = {
                    "login": author_login,
                    "name": author_name or author_login,
                    "avatar_url": avatar_url,
                    "commit_count": 0,
                    "last_commit_date": date_str,
                }
            author_map[author_login]["commit_count"] += 1

        top_contributors = sorted(author_map.values(), key=lambda x: x["commit_count"], reverse=True)

        recent_commits = []
        for c in commits[:10]:
            commit_author = c.get("author", {}).get("login") if c.get("author") else c.get("commit", {}).get("author", {}).get("name", "unknown")
            msg = c.get("commit", {}).get("message", "").split("\n")[0]
            recent_commits.append({
                "sha": (c.get("sha") or "")[:7],
                "message": msg,
                "author": commit_author,
                "date": c.get("commit", {}).get("author", {}).get("date", ""),
                "html_url": c.get("html_url", ""),
            })

        return {
            "timeframe_days": days,
            "total_commits": len(commits),
            "commits_last_7_days": commits_last_7_days,
            "commits_in_timeframe": commits_in_timeframe,
            "active_contributors_count": len(top_contributors),
            "top_contributors": top_contributors,
            "recent_commits": recent_commits,
        }
    except Exception as e:
        return {"error": f"Failed to fetch contributor activity: {e}"}


async def fetch_github_release_and_ci_status_async(owner: str, repo: str, token: str = "") -> dict:
    """Fetch latest release tags and recent GitHub Actions CI workflow run health."""
    if not token:
        token = await get_github_access_token_async()

    if not owner or not repo:
        return {"error": "Missing repository owner or repo name."}
    if not token:
        return {"error": "GitHub is not linked or unauthorized. Please connect your GitHub account."}

    rel_url = f"https://api.github.com/repos/{owner}/{repo}/releases?per_page=5"
    runs_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs?per_page=10"
    headers = _get_github_headers(token)

    releases = []
    workflow_runs = []

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            rel_task = client.get(rel_url, headers=headers)
            runs_task = client.get(runs_url, headers=headers)
            rel_resp, runs_resp = await asyncio.gather(rel_task, runs_task, return_exceptions=True)

            if not isinstance(rel_resp, Exception) and rel_resp.status_code == 200:
                for r in rel_resp.json():
                    releases.append({
                        "id": r.get("id"),
                        "tag_name": r.get("tag_name"),
                        "name": r.get("name") or r.get("tag_name"),
                        "published_at": r.get("published_at"),
                        "prerelease": r.get("prerelease", False),
                        "html_url": r.get("html_url", ""),
                    })

            if not isinstance(runs_resp, Exception) and runs_resp.status_code == 200:
                for w in runs_resp.json().get("workflow_runs", []):
                    workflow_runs.append({
                        "id": w.get("id"),
                        "name": w.get("name") or "CI Workflow",
                        "head_branch": w.get("head_branch"),
                        "status": w.get("status"),  # completed, in_progress, queued
                        "conclusion": w.get("conclusion"),  # success, failure, cancelled
                        "created_at": w.get("created_at"),
                        "html_url": w.get("html_url", ""),
                    })

        passing = sum(1 for w in workflow_runs if w.get("conclusion") == "success")
        failing = sum(1 for w in workflow_runs if w.get("conclusion") in ["failure", "timed_out"])
        in_progress = sum(1 for w in workflow_runs if w.get("status") in ["in_progress", "queued"])

        return {
            "latest_release": releases[0] if releases else None,
            "releases": releases,
            "recent_workflow_runs": workflow_runs,
            "ci_health": {
                "total_tracked": len(workflow_runs),
                "passing_runs": passing,
                "failing_runs": failing,
                "in_progress_runs": in_progress,
                "is_healthy": failing == 0,
            },
        }
    except Exception as e:
        return {"error": f"Failed to fetch release and CI status: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB LANGCHAIN @tool DESCRIPTORS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_pull_requests_summary(owner: str, repo: str, token: str = "") -> dict:
    """Fetch GitHub PR review status, stale PRs (>48h), draft PRs, and merged PRs for the project repo."""
    import asyncio
    return asyncio.run(fetch_github_pull_requests_async(owner, repo, token))


@tool
def get_issues(owner: str, repo: str, token: str = "") -> dict:
    """Fetch GitHub issues summary (open, closed, unassigned, and blocker bugs)."""
    import asyncio
    return asyncio.run(fetch_github_issues_async(owner, repo, token))


@tool
def get_contributor_activity(owner: str, repo: str, token: str = "", days: int = 14) -> dict:
    """Fetch developer velocity, commit trends over last 7/14 days, and active team contributors."""
    import asyncio
    return asyncio.run(fetch_github_contributor_activity_async(owner, repo, token, days=days))


@tool
def get_release_and_ci_status(owner: str, repo: str, token: str = "") -> dict:
    """Fetch latest GitHub release tags and GitHub Actions CI/CD workflow health."""
    import asyncio
    return asyncio.run(fetch_github_release_and_ci_status_async(owner, repo, token))



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

# All Read-Only Analytics & Insights Tools (Includes sprint insights)
ANALYST_TOOLS = [
    get_user_standup,
    get_tasks_summary,
    get_issues_summary,
    get_member_workload,
    get_sprint_insights,
]

ALL_TOOLS = DBWRITE_TOOLS + ANALYST_TOOLS


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
        "name": "jira_issue_management",
        "title": "Jira Issue Sync, Search & Safe Ticket Creation",
        "connectorId": "jira",
        "description": "Discovers Atlassian cloudId and resources, searches active or uncompleted Jira issues using universal JQL, and safely creates Jira tasks/issues with project key and verification.",
        "content": """---
name: jira_issue_management
title: Jira Issue Sync, Search & Safe Ticket Creation
description: Discovers Atlassian cloudId and resources, searches active or uncompleted Jira issues using universal JQL, and safely creates Jira tasks/issues with project key and verification.
connectorId: jira
createdBy: default
version: 1.0.0
---

# Jira Issue Management Skill

## Context & Objectives
This skill instructs Kaya and MCP sub-agents on how to interact with Atlassian Jira: dynamically discovering accessible sites and cloud IDs, querying active/uncompleted issues via adaptive JQL, safely creating new tasks or issues, and deduplicating against workspace project tasks.

## Discovery & Prerequisites Protocol
1. Never guess the Cloud ID: Jira requires a valid Atlassian cloudId. Always run getAccessibleAtlassianResources first to resolve the active cloudId and site URL.
2. Never assume fixed Project Keys: If the user prompt mentions a project key (e.g. KAN, PROJ), use it. Otherwise, extract the project key from issues returned by JQL search.

## Step-by-Step Tool Execution Workflow

### Step 1 — Resource & Cloud ID Discovery:
- Tool: getAccessibleAtlassianResources
- Arguments: {}
- Capture: cloudId (e.g. e56da97a-1da4-40fd-bed2-4c2663ef28e7) and url from data.resources[0].

### Step 2 — Query / Search Issues (Get Tasks):
- Tool: searchJiraIssuesUsingJql
- Default Universal JQL (Find active/uncompleted tasks):
```json
{
  "cloudId": "<discovered_cloudId>",
  "jql": "statusCategory != Done ORDER BY updated DESC",
  "maxResults": 20
}
```
*(Note: If statusCategory != Done returns no items or errors, fall back to "status IS NOT NULL ORDER BY updated DESC").*
- If User Specified a Project:
  jql: "project = '<ProjectKey>' AND statusCategory != Done ORDER BY updated DESC"
- Extract Fields: key, summary, status.name, priority.name, project.key, assignee.displayName.
- Deduplication Check: Compare retrieved Jira task summaries and keys against internal workspace project tasks to identify new or missing tasks to suggest or bring over.

### Step 3 — Safe Task / Issue Creation (Create Tasks):
- Tool: createJiraIssue
- Arguments:
```json
{
  "cloudId": "<discovered_cloudId>",
  "projectKey": "<projectKey>",
  "summary": "<Concise, actionable task summary>",
  "issueType": "Task",
  "description": "<Detailed context, acceptance criteria, or source reference>"
}
```
- Read-After-Write Verification: Confirm that the response contains the newly created issue key and id.

## Strict Anti-Hallucination & Execution Rules
- No Speculative Creation: Never tell the user an issue has been created until createJiraIssue returns a valid payload.
- Never Restrict to Current User: Do not include assignee = currentUser() in JQL unless the user explicitly requested only their own tickets.
- No Fabricated Keys or Links: Verbatim quote Jira keys (e.g. KAN-1) and use official URLs.
- Workspace Deduplication: Always cross-reference against project tasks before creating or suggesting tasks to bring into this workspace.

## Finalization Handshake
End execution by calling finalize_result with data, resolved_context, and summary.

## Execution Configuration
```json
{
  "maxSteps": 3,
  "pinnedTools": {
    "jira": [
      {
        "name": "getAccessibleAtlassianResources",
        "description": "Discover accessible Atlassian cloudId and resources. Always call first before running JQL or creating issues.",
        "inputSchema": { "type": "object", "properties": {}, "required": [] }
      },
      {
        "name": "searchJiraIssuesUsingJql",
        "description": "Search Jira issues using JQL. Pass cloudId and bounded jql.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "cloudId": { "type": "string" },
            "jql": { "type": "string" },
            "maxResults": { "type": "number" }
          },
          "required": ["cloudId", "jql"]
        }
      },
      {
        "name": "createJiraIssue",
        "description": "Create a new Jira issue/task. Requires cloudId, projectKey, summary, and issueType.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "cloudId": { "type": "string" },
            "projectKey": { "type": "string" },
            "summary": { "type": "string" },
            "issueType": { "type": "string" },
            "description": { "type": "string" }
          },
          "required": ["cloudId", "projectKey", "summary"]
        }
      }
    ]
  }
}
```""",
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
            merged = list(skills)
            for default_s in DEFAULT_SKILLS_FALLBACK:
                if not any(s.get("name") == default_s.get("name") for s in merged):
                    merged.append(default_s)
            print(f"[FETCH SKILLS] [OK] Retrieved {len(skills)} skill(s) from Convex DB (total {len(merged)} active with defaults).")
            return merged
    except Exception as e:
        safe_e = str(e).encode("ascii", "replace").decode("ascii")
        print(f"[FETCH SKILLS] Notice: Could not fetch skills from Convex endpoint ({safe_e}), using default skills catalog.")
    return DEFAULT_SKILLS_FALLBACK

