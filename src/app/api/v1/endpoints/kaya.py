import asyncio
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, Request, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.agents.graph.kaya_graph import kaya_graph
from app.core.utils import (
    message_chunk_event,
    checkpoint_event,
    interrupt_event,
    custom_event,
    error_event,
)
from langchain_core.messages import AIMessageChunk

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

    if request_type == "run":
        graph_input = body.get("state") or {}
        graph_input["user_id"] = user_id
        graph_input["user_name"] = user_name
        graph_input["project_id"] = project_id

    elif request_type == "resume":
        resume_value = body.get("resume", "cancel")
        graph_input = Command(resume=resume_value)

    elif request_type == "fork":
        cfg = body.get("config")
        if not cfg:
            raise HTTPException(status_code=400, detail="config is required for fork")
        fork_state = body.get("state", None)
        config = await kaya_graph.aupdate_state(cfg, fork_state)
        graph_input = None

    elif request_type == "replay":
        cfg = body.get("config")
        if not cfg:
            raise HTTPException(status_code=400, detail="config is required for replay")
        graph_input = None

    else:
        user_message = body.get("message")
        if user_message:
            graph_input = {
                "messages": [HumanMessage(content=user_message)],
                "user_id": user_id,
                "user_name": user_name,
                "project_id": project_id,
                "thread_id": thread_id,
            }
        else:
            graph_input = body.get("state", None)

    stop_event = asyncio.Event()
    active_connections[thread_id] = stop_event

    async def generate_events() -> AsyncGenerator[dict, None]:
        try:
            yield custom_event({"status": "connected", "thread_id": thread_id, "agent": "kaya"})

            async for chunk in kaya_graph.astream(
                graph_input,
                config,
                stream_mode=["debug", "messages", "updates", "custom"],
            ):
                if stop_event.is_set():
                    break

                chunk_type, chunk_data = chunk

                if chunk_type == "custom":
                    yield custom_event(chunk_data)

                elif chunk_type == "debug":
                    debug_type = chunk_data.get("type")
                    if debug_type == "checkpoint":
                        yield checkpoint_event(chunk_data["payload"])
                    elif debug_type == "task_result":
                        interrupts = chunk_data["payload"].get("interrupts", [])
                        if interrupts:
                            yield interrupt_event(interrupts)

                elif chunk_type == "messages":
                    msg, metadata = chunk_data
                    node_name = metadata.get("langgraph_node", "unknown")

                    if isinstance(msg, AIMessageChunk):
                        has_content = bool(msg.content or msg.tool_call_chunks)
                        if has_content:
                            yield message_chunk_event(node_name, msg)

            # Mark completion
            yield custom_event({"status": "completed", "thread_id": thread_id, "agent": "kaya"})

        except Exception as e:
            print(f"[/kaya STREAM ERROR] {e}")
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
