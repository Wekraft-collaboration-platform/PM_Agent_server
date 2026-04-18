# utils.py — SSE event formatters.

import json
from langchain_core.messages import AIMessageChunk
from langgraph.types import StateSnapshot


def message_chunk_event(node_name: str, message_chunk) -> dict:
    print(
        f"[message_chunk_event] node={node_name}, msg_id={getattr(message_chunk, 'id', None)}"
    )
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
        print(
            f"[message_chunk_event] content_len={len(content)}, tool_call_chunks={len(tool_call_chunks)}, tool_calls={len(tool_calls)}"
        )

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
    print(f"[checkpoint_event] Formatting checkpoint snapshot")
    data = format_state_snapshot(snapshot)
    print(
        f"[checkpoint_event] next={data.get('next')}, keys_in_values={list(data.get('values', {}).keys())}"
    )
    return {
        "event": "checkpoint",
        "data": json.dumps(data),
    }


def interrupt_event(interrupts: list) -> dict:
    print(f"[interrupt_event] {len(interrupts)} interrupt(s) received")
    formatted = []
    for i in interrupts:
        # LangGraph 'debug' mode sends interrupts as dicts with 'value' and 'id'
        if isinstance(i, dict) and "value" in i:
            value = i["value"]
            interrupt_id = i.get("id")
        else:
            # Handle list of Interrupt objects
            value = getattr(i, "value", i)
            interrupt_id = getattr(i, "id", None)

        print(f"[interrupt_event] interrupt value={value}, id={interrupt_id}")
        formatted.append({"value": value, "id": interrupt_id})

    return {
        "event": "interrupt",
        "data": json.dumps(formatted),
    }


def custom_event(data: dict) -> dict:
    print(f"[custom_event] data={data}")
    return {
        "event": "custom",
        "data": json.dumps(data),
    }


def error_event(message: str) -> dict:
    print(f"[error_event] ERROR: {message}")
    return {
        "event": "error",
        "data": json.dumps({"error": message}),
    }


def format_state_snapshot(snapshot):
    print(f"[format_state_snapshot] Formatting snapshot type={type(snapshot).__name__}")

    def get_val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    values = get_val(snapshot, "values", {})
    print(
        f"[format_state_snapshot] values keys={list(values.keys()) if isinstance(values, dict) else 'not a dict'}"
    )

    serialized_values = {}
    for key, val in values.items():
        if key == "messages" and isinstance(val, list):
            print(f"[format_state_snapshot] Serializing {len(val)} messages")
            serialized_values[key] = [_serialize_message(m) for m in val]
        else:
            serialized_values[key] = val

    next_nodes = get_val(snapshot, "next", [])
    config = get_val(snapshot, "config", {})
    metadata = get_val(snapshot, "metadata", {})
    parent_config = get_val(snapshot, "parent_config")
    print(
        f"[format_state_snapshot] next={list(next_nodes) if next_nodes else []}, metadata={metadata}"
    )

    interrupts = []
    tasks = get_val(snapshot, "tasks", [])
    print(f"[format_state_snapshot] {len(tasks)} task(s) found")
    for task in tasks:
        task_interrupts = get_val(task, "interrupts", [])
        for interrupt in task_interrupts:
            value = get_val(interrupt, "value", interrupt)
            print(f"[format_state_snapshot] Task interrupt found: value={value}")
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
    original_type = msg.__class__.__name__

    # Map LangChain class names to standard lowercase types used by frontend
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

    print(f"[_serialize_message] mapped {original_type} -> {type_str}")

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
