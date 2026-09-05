import asyncio
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, Request, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.core.utils import (
    message_chunk_event,
    checkpoint_event,
    interrupt_event,
    custom_event,
    error_event,
)

router = APIRouter(prefix="/kaya", tags=["Kaya AI Agent"])

# Track active SSE stream connections
active_connections: dict[str, asyncio.Event] = {}


@router.post("")
@router.post("/")
async def kaya_agent_endpoint(request: Request):
    """Primary execution & streaming endpoint for Kaya AI Agent."""
    body: dict = await request.json()
    print(f"[/kaya] Received request: {body}")

    thread_id: str = body.get("thread_id") or str(uuid.uuid4())
    user_id: str = body.get("user_id") or body.get("state", {}).get("user_id")
    user_name: str = body.get("user_name") or body.get("state", {}).get("user_name")
    project_id: str = body.get("project_id") or body.get("state", {}).get("project_id")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    request_type: str = body.get("type", "message")
    model: str = body.get("model") or "fast"
    print(f"[/kaya] thread_id={thread_id}, type={request_type}, model={model}")

    config = {"configurable": {"thread_id": thread_id, "model": model}}

    stop_event = asyncio.Event()
    active_connections[thread_id] = stop_event

    async def generate_events() -> AsyncGenerator[dict, None]:
        try:
            # Yield initial connection confirmation
            yield custom_event({"status": "connected", "thread_id": thread_id, "agent": "kaya"})

            # Note: graph.astream will stream when graph module is attached
            yield custom_event({"status": "ready", "agent": "kaya"})

        except Exception as e:
            yield error_event(str(e))
        finally:
            active_connections.pop(thread_id, None)

    return EventSourceResponse(generate_events())


@router.post("/stop")
async def stop_kaya_stream(request: Request):
    """Stop active execution stream for a given thread_id."""
    body = await request.json()
    thread_id = body.get("thread_id")
    if not thread_id or thread_id not in active_connections:
        return {"status": "not_running", "thread_id": thread_id}

    active_connections[thread_id].set()
    return {"status": "stopped", "thread_id": thread_id}
