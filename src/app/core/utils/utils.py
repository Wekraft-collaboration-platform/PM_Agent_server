# utils.py — SSE event formatters matching agent_temp structure.

import json
from langchain_core.messages import AIMessageChunk


def message_chunk_event(node_name: str, message_chunk) -> dict:
    """Format AIMessageChunk into standard SSE event dictionary."""
    content = ""
    tool_call_chunks = []
    tool_calls = []

    if isinstance(message_chunk, AIMessageChunk):
        content = (
            message_chunk.content if isinstance(message_chunk.content, str) else ""
        )
        tool_call_chunks = [
            {
                "name": t.get("name"),
                "args": t.get("args", ""),
                "id": t.get("id"),
                "index": t.get("index"),
            }
            for t in (message_chunk.tool_call_chunks or [])
        ]
        tool_calls = [
            {"name": t.get("name"), "args": t.get("args", {}), "id": t.get("id")}
            for t in (getattr(message_chunk, "tool_calls", None) or [])
        ]

    return {
        "event": "message_chunk",
        "data": json.dumps(
            {
                "node_name": node_name,
                "message_chunk": {
                    "content": content,
                    "id": getattr(message_chunk, "id", None),
                    "tool_calls": tool_calls,
                    "tool_call_chunks": tool_call_chunks,
                },
            }
        ),
    }


def checkpoint_event(snapshot) -> dict:
    """Format LangGraph state snapshot into SSE checkpoint event."""
    data = format_state_snapshot(snapshot)
    return {
        "event": "checkpoint",
        "data": json.dumps(data),
    }


def interrupt_event(interrupts: list) -> dict:
    """Format LangGraph interrupts into SSE interrupt event."""
    formatted = []
    for i in interrupts:
        if isinstance(i, dict) and "value" in i:
            value = i["value"]
            interrupt_id = i.get("id")
        else:
            value = getattr(i, "value", i)
            interrupt_id = getattr(i, "id", None)

        formatted.append({"value": value, "id": interrupt_id})

    return {
        "event": "interrupt",
        "data": json.dumps(formatted),
    }


def custom_event(data: dict) -> dict:
    """Format custom data payload into SSE custom event."""
    print(f"[custom_event] data={data}")
    return {
        "event": "custom",
        "data": json.dumps(data),
    }


def error_event(message: str) -> dict:
    """Format error message into SSE error event."""
    print(f"[error_event] ERROR: {message}")
    return {
        "event": "error",
        "data": json.dumps({"error": message}),
    }


def format_state_snapshot(snapshot):
    """Serialize state snapshot values and message history into JSON-serializable dictionary."""
    def get_val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    values = get_val(snapshot, "values", {})

    serialized_values = {}
    for key, val in values.items():
        if key == "messages" and isinstance(val, list):
            serialized_values[key] = [_serialize_message(m) for m in val]
        else:
            serialized_values[key] = val

    next_nodes = get_val(snapshot, "next", [])
    config = get_val(snapshot, "config", {})
    metadata = get_val(snapshot, "metadata", {})
    parent_config = get_val(snapshot, "parent_config")

    interrupts = []
    tasks = get_val(snapshot, "tasks", [])
    for task in tasks:
        task_interrupts = get_val(task, "interrupts", [])
        for interrupt in task_interrupts:
            value = get_val(interrupt, "value", interrupt)
            interrupts.append({"value": value})

    return {
        "values": serialized_values,
        "next": list(next_nodes) if next_nodes else [],
        "config": config,
        "interrupts": interrupts,
        "parent_config": parent_config,
        "metadata": metadata,
    }


def _serialize_message(msg) -> dict:
    """Serialize LangChain BaseMessage objects into JSON dict for frontend."""
    original_type = msg.__class__.__name__

    if "Human" in original_type:
        type_str = "human"
    elif "AI" in original_type:
        type_str = "ai"
    elif "Tool" in original_type:
        type_str = "tool"
    elif "System" in original_type:
        type_str = "system"
    else:
        type_str = original_type.lower().replace("message", "")

    base: dict = {
        "type": type_str,
        "content": msg.content if isinstance(msg.content, str) else str(msg.content),
        "id": getattr(msg, "id", None),
    }
    if getattr(msg, "tool_calls", None):
        base["tool_calls"] = [
            {"name": t.get("name"), "args": t.get("args", {}), "id": t.get("id")}
            for t in msg.tool_calls
        ]
    if getattr(msg, "tool_call_id", None):
        base["tool_call_id"] = msg.tool_call_id
        base["name"] = getattr(msg, "name", None)
    return base
