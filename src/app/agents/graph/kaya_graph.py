import os
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR ROUTER DECISION SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorDecision(BaseModel):
    actions: List[str] = Field(
        description=(
            "List of sub-agent actions to trigger in parallel (can select 1 or multiple):\n"
            "- 'db_write': for user memory search, calendar events, daily user standup, or report scheduler\n"
            "- 'analyst': for task/issue summaries, member workloads, project insights, or PRD bulk task/issue creation\n"
            "- 'sprint': for sprint insights, sprint creation, or backlog item assignments\n"
            "- 'direct_response': if query is a simple greeting or general conversation requiring no tools"
        )
    )
    reasoning: str = Field(description="Reasoning for routing decision.")


ROUTER_SYSTEM_PROMPT = """
You are the primary Supervisor Router for the WEKRAFT AI Platform.
Your sole job is to analyze incoming user queries and decide which specialized sub-agents to trigger:

Available Sub-Agent Actions:
1. 'db_write': All state mutation / write operations: User long-term memory search (Mem0), calendar scheduling, report scheduler setup, or bulk creation of tasks/issues from PRDs/documents.
2. 'analyst': All read-only analytics: User daily standup, project task summary, active/critical issue tracking, team member workload analysis, or project insights/timeline.
3. 'sprint': Sprint insights/velocity, sprint creation, or backlog task assignment UI triggers.
4. 'direct_response': Simple greetings (hi, hello, how are you), general chatter, or casual questions that require no database/tool queries.

Instructions:
- You may select MULTIPLE sub-agents if the user query asks for multiple items (e.g. 'Show my standup and sprint velocity' -> ['analyst', 'sprint']).
- Always provide clear concise reasoning.
"""



async def route_user_request(user_input: str, project_id: Optional[str] = None) -> SupervisorDecision:
    """
    Evaluates user input using Groq LLM router and outputs SupervisorDecision.
    Only prints final router results.
    """
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    router_model = os.getenv("GROQ_ROUTER_MODEL", "openai/gpt-oss-120b")

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"User Query: '{user_input}' | Active Project ID: {project_id or 'none'}")
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

        # Only print final clean router results
        print(f"[ROUTER DECISION] Actions: {decision.actions} | Reasoning: {decision.reasoning}")
        return decision

    except Exception as e:
        fallback = SupervisorDecision(
            actions=["direct_response"],
            reasoning=f"Fallback routing due to LLM error: {e}"
        )
        print(f"[ROUTER DECISION] Actions: {fallback.actions} | Reasoning: {fallback.reasoning}")
        return fallback
