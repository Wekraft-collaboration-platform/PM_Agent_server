"""
test_router_skills.py — Verification of Groq gpt-oss-120b Router for Sub-Agent & Skill Selection.

Tests 5 complex real-world queries against Level 1 Skill Metadata Catalog.
"""

import os
import sys
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1. LEVEL 1 SKILLS CATALOG (Lightweight Metadata ~50 tokens each)
# ─────────────────────────────────────────────────────────────────────────────

SKILLS_CATALOG = [
    {
        "name": "sentry_error_triage",
        "title": "Sentry Error Triage & Root Cause Analysis",
        "connector": "sentry",
        "description": "Discovers Sentry organizations, fetches unresolved production errors with bounded queries, identifies culprit routes, and isolates critical system crashes.",
    },
    {
        "name": "linear_issue_management",
        "title": "Linear Issue Sync & Safe Ticket Creation",
        "connector": "linear",
        "description": "Discovers Linear workspace teams, lists active/backlog tickets without truncation, safely creates tickets with required team UUIDs, and verifies creation.",
    },
    {
        "name": "incident_escalation_triage",
        "title": "Cross-App Production Incident Escalation",
        "connector": "sentry, linear, project",
        "description": "Coordinates Sentry telemetry, Linear ticketing, and internal project issues to escalate critical production crashes (>10 events) into actionable engineering tickets.",
    },
]

# Format Level 1 Catalog for Prompt Injection
CATALOG_PROMPT = "\n".join(
    f"- Skill: '{s['name']}' (Connectors: {s['connector']})\n  Description: {s['description']}"
    for s in SKILLS_CATALOG
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRUCTURED DECISION SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorSkillDecision(BaseModel):
    actions: List[str] = Field(
        description=(
            "List of sub-agent actions to trigger in parallel:\n"
            "- 'mcp': External 3rd-party SaaS integrations ONLY (Linear, Sentry, Jira, Vercel, Slack).\n"
            "- 'analyst': Internal project read analytics (tasks, issues, standups, workload).\n"
            "- 'db_write': Internal project mutations (creating/updating tasks or issues in this project).\n"
            "- 'sprint': Internal sprint velocity, sprint planning, and active sprint items.\n"
            "- 'direct_response': Greetings or general PM chat without tools."
        )
    )
    selected_skill: Optional[str] = Field(
        default=None,
        description=(
            "Name of the SINGLE best matching skill from the Available Skills Catalog, "
            "or null if no specialized procedural skill is required."
        )
    )
    reasoning: str = Field(description="Brief explanation of the sub-agent and skill routing choice.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROUTER PROMPT
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = f"""You are the Master Supervisor Router for the WEKRAFT AI Platform.
Analyze the user's incoming query and make two decisions:
1. Select the exact sub-agents required ('actions').
2. Select the SINGLE most appropriate procedural skill ('selected_skill') from the Available Skills Catalog, or null if none match.

Available Sub-Agents:
- 'mcp': External tools & integrations (Linear, Sentry, Vercel, Jira, Slack).
- 'analyst': Read internal project data (tasks, issues, standups, deadlines).
- 'db_write': Write/create internal project tasks or issues.
- 'sprint': Sprint management and velocity.
- 'direct_response': Casual conversation.

Available Skills Catalog (Level 1 Metadata):
{CATALOG_PROMPT}

Rules:
- If a query needs external tools, select 'mcp'.
- If a query asks to inspect internal project tasks/issues, select 'analyst'.
- If a query asks to create tasks/issues locally in this project, select 'db_write'.
- Choose at most ONE skill that provides the authoritative playbook for the task.
"""


async def run_test():
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    router_model = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b")

    llm = ChatOpenAI(
        model=router_model,
        openai_api_key=groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.0,
    ).with_structured_output(SupervisorSkillDecision)

    test_queries = [
        {
            "id": 1,
            "title": "Linear & Internal Project Diff & Creation",
            "query": "check my current issues in project and not closed or done project in linear , and if we miss any , create those isues here in this project.",
            "expected_actions": ["analyst", "mcp", "db_write"],
            "expected_skill": ["linear_issue_management", "incident_escalation_triage"],
        },
        {
            "id": 2,
            "title": "Vercel + Sentry Deployments & Task Deduplication",
            "query": "check deployment status of the projet in vercel , any error , check sentry also , and then only suggest top 5 task to create , and check in my tasks also if they are already present and not completed !",
            "expected_actions": ["mcp", "analyst"],
            "expected_skill": ["sentry_error_triage", None],
        },
        {
            "id": 3,
            "title": "Sentry Production Crash Triage",
            "query": "Find all unresolved production errors in Sentry for this project, identify the top crash routes, and show me the affected users.",
            "expected_actions": ["mcp"],
            "expected_skill": ["sentry_error_triage"],
        },
        {
            "id": 4,
            "title": "Internal Sprint Velocity & Standup (Zero MCP)",
            "query": "Analyze our active sprint velocity, compare with member standup blockers, and summarize team workload.",
            "expected_actions": ["analyst", "sprint"],
            "expected_skill": [None],
        },
        {
            "id": 5,
            "title": "Cross-App Production Incident Escalation to Linear",
            "query": "We have critical production errors in Sentry (>10 events). Check if they exist in Linear, and escalate the untracked ones as Urgent Linear tickets.",
            "expected_actions": ["mcp"],
            "expected_skill": ["incident_escalation_triage", "linear_issue_management"],
        },
    ]

    print("=" * 80)
    print(f"RUNNING ROUTER TEST WITH GROQ: {router_model}")
    print("=" * 80)

    for tc in test_queries:
        print(f"\n--- TEST #{tc['id']}: {tc['title']} ---")
        print(f"Query: \"{tc['query']}\"")

        messages = [
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=tc["query"]),
        ]

        result: SupervisorSkillDecision = await llm.ainvoke(messages)

        print(f"-> Selected Actions : {result.actions}")
        print(f"-> Selected Skill   : {result.selected_skill}")
        print(f"-> Router Reasoning : {result.reasoning}")

        # Check actions match
        actions_ok = all(a in result.actions for a in tc["expected_actions"])
        skill_ok = result.selected_skill in tc["expected_skill"]

        print(f"-> Sub-Agents Match : {'[PASS]' if actions_ok else '[FAIL]'}")
        print(f"-> Skill Match      : {'[PASS]' if skill_ok else '[FAIL]'}")


if __name__ == "__main__":
    asyncio.run(run_test())
