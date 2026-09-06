import asyncio
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, Request, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from sse_starlette.sse import EventSourceResponse

from app.agents.graph.kaya_graph import kaya_graph
from app.core.guardrails.input_guardrails import input_guardrails
from app.core.utils import (
    message_chunk_event,
    checkpoint_event,
    interrupt_event,
    custom_event,
    error_event,
)
from langchain_core.messages import AIMessageChunk, SystemMessage, AIMessage

BLOCKED_MESSAGE = "Sorry this request failed to pass as it dosent seems appropriate. Try sending again."

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
    project_name: str = body.get("project_name") or body.get("state", {}).get("project_name")

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    request_type: str = body.get("type", "message")
    model: str = body.get("model") or "fast"
    print(f"[/kaya] thread_id={thread_id}, type={request_type}, model={model}")

    config = {"configurable": {"thread_id": thread_id, "model": model}}

    def _normalize_messages(raw_msgs):
        if not raw_msgs:
            return []
        converted = []
        for m in raw_msgs:
            if isinstance(m, dict):
                m_type = m.get("type") or m.get("role")
                m_content = m.get("content", "")
                if m_type in ["user", "human"]:
                    converted.append(HumanMessage(content=m_content))
                elif m_type in ["ai", "assistant"]:
                    converted.append(AIMessage(content=m_content))
                elif m_type == "system":
                    converted.append(SystemMessage(content=m_content))
                else:
                    converted.append(HumanMessage(content=str(m_content)))
            else:
                converted.append(m)
        return converted

    is_blocked = False
    guardrail_reason = ""

    if request_type == "run":
        graph_input = body.get("state") or {}
        graph_input["user_id"] = user_id
        graph_input["user_name"] = user_name
        graph_input["project_id"] = project_id
        graph_input["project_name"] = project_name
        graph_input["thread_id"] = thread_id
        if "messages" in graph_input:
            graph_input["messages"] = _normalize_messages(graph_input["messages"])


    elif request_type == "resume":
        resume_value = body.get("resume", "cancel")
        graph_input = Command(resume=resume_value)

    elif request_type == "fork":
        cfg = body.get("config")
        if not cfg:
            raise HTTPException(status_code=400, detail="config is required for fork")
        fork_state = body.get("state", None)
        if fork_state and "messages" in fork_state:
            fork_state["messages"] = _normalize_messages(fork_state["messages"])
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
            if graph_input and "messages" in graph_input:
                graph_input["messages"] = _normalize_messages(graph_input["messages"])

    # ── Run Input Guardrails on Latest User Message ──
    if isinstance(graph_input, dict) and "messages" in graph_input and graph_input["messages"]:
        latest_user_text = ""
        for m in reversed(graph_input["messages"]):
            if isinstance(m, HumanMessage):
                latest_user_text = m.content if isinstance(m.content, str) else str(m.content)
                break

        if latest_user_text:
            guardrail_res = await input_guardrails.run_input_guardrails(latest_user_text)
            is_safe = guardrail_res.get("is_safe", True)
            risk_score = guardrail_res.get("risk_score", 0.0)
            guardrail_reason = guardrail_res.get("reason", "")
            print(f"[INPUT GUARDRAIL RESULT] Safe: {is_safe} | Risk Score: {risk_score} | Reason: {guardrail_reason}")

            if not is_safe:
                is_blocked = True
            else:
                # Apply redacted text if any PII was sanitized
                if guardrail_res.get("redacted_text") and graph_input["messages"]:
                    for idx in range(len(graph_input["messages"]) - 1, -1, -1):
                        if isinstance(graph_input["messages"][idx], HumanMessage):
                            graph_input["messages"][idx] = HumanMessage(content=guardrail_res["redacted_text"])
                            break

    stop_event = asyncio.Event()
    active_connections[thread_id] = stop_event

    async def generate_events() -> AsyncGenerator[dict, None]:
        try:
            yield custom_event({"status": "connected", "thread_id": thread_id, "agent": "kaya"})

            # If input was flagged as unsafe, emit a synthetic checkpoint first so the client shows the user query + reply
            if is_blocked:
                user_msgs = graph_input.get("messages", []) if isinstance(graph_input, dict) else []
                synthetic_snapshot = {
                    "values": {
                        "messages": user_msgs,
                        "user_id": user_id,
                        "user_name": user_name,
                        "project_id": project_id,
                        "project_name": project_name,
                    },
                    "next": ["kaya_direct_node"],
                    "config": {"configurable": {"checkpoint_id": str(uuid.uuid4()), "thread_id": thread_id}},
                    "metadata": {"step": 1, "writes": {"kaya_direct_node": {"messages": []}}},
                }
                yield checkpoint_event(synthetic_snapshot)

                yield custom_event({"agent_status": "Kaya reviewed your request."})
                blocked_chunk = AIMessageChunk(
                    content=BLOCKED_MESSAGE,
                    id=str(uuid.uuid4()),
                )
                yield message_chunk_event("kaya_direct_node", blocked_chunk)
                yield custom_event({"status": "completed", "thread_id": thread_id, "agent": "kaya"})
                return

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
                    node_name = metadata.get("langgraph_node", "kaya_direct_node")

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
