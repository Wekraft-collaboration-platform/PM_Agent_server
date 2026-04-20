# graph.py — Kaya AI

import os
from datetime import datetime
import httpx

from langchain_core.messages import SystemMessage, ToolMessage
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
class KayaState(MessagesState):
    user_id: str
    thread_id: str
    project_id: str | None


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
_llm = ChatOpenAI(
    model=os.getenv("KAYA_MODEL", "gpt-4.1-nano"),
    temperature=0.3,
    streaming=True,
).bind_tools([create_calendar_event])

_mem0 = MemoryClient()


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

    response = _llm.invoke(full_messages)

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
        if tool_call["name"] == "create_calendar_event":
            sends.append(Send("tools", tool_call))

    return sends if sends else END


# ─────────────────────────────────────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────────────────────────────────────
checkpointer = InMemorySaver()


def build_graph():
    g = StateGraph(KayaState)

    g.add_node("kaya", kaya)
    g.add_node("tools", tools)

    g.add_edge(START, "kaya")
    g.add_conditional_edges("kaya", assign_tool)
    g.add_edge("tools", "kaya")

    return g.compile(checkpointer=checkpointer)


graph = build_graph()
print("[graph] Kaya agent graph compiled successfully")
