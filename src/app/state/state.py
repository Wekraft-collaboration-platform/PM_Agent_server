"""
state.py — Centralized Shared State for WEKRAFT Agent & Sub-Agents.

This shared state acts as the unified memory and context bus across the primary Kaya graph,
the Supervisor Router, and all specialized sub-agents (Analyst, DB Write, Sprint).
"""

from typing import Annotated, List, Dict, Any, Optional, Union
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState


# ─────────────────────────────────────────────────────────────────────────────
# SMART REDUCER FOR SUB-AGENT ISOLATED THREADS
# ─────────────────────────────────────────────────────────────────────────────

RESET_SENTINEL = "__RESET__"


def reset_or_add(old: Optional[List[Any]], new: Optional[List[Any]]) -> List[Any]:
    """
    State reducer for sub-agent private message channels.
    If the incoming list starts with RESET_SENTINEL, the state resets (clears past buffer).
    Otherwise, new messages are appended to the existing thread.
    """
    old_list = old or []
    new_list = new or []
    if new_list and new_list[0] == RESET_SENTINEL:
        return new_list[1:]  # Start fresh with new list (dropping reset marker)
    return old_list + new_list


# ─────────────────────────────────────────────────────────────────────────────
# CENTRAL SHARED STATE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

class SupervisorState(TypedDict):
    """
    Central shared state used by the Supervisor Graph, Kaya Agent, and Sub-Agents.
    """

    # 1. Main Conversation History (Standard LangGraph message reducer)
    messages: Annotated[List[BaseMessage], add_messages]

    # 2. Session & User Identifiers
    user_id: str
    user_name: Optional[str]
    project_id: Optional[str]
    project_name: Optional[str]
    thread_id: Optional[str]

    # 3. Router Execution Control (Single or Parallel Multi-Agent Fan-Out)
    next: Union[str, List[str]]
    action_type: Optional[str]
    router_reasoning: Optional[str]

    # 4. Isolated Sub-Agent Message Buffers (Smart Reducers)
    _analyst_messages: Annotated[List[Any], reset_or_add]
    _analyst_tool_call_id: Optional[str]
    _db_write_messages: Annotated[List[Any], reset_or_add]
    _sprint_messages: Annotated[List[Any], reset_or_add]
    _mcp_messages: Annotated[List[Any], reset_or_add]
    _mcp_tool_call_id: Optional[str]

    # 5. Shared Memory & Context Bus (Mem0, Semantic Cache, Tool Outputs, Document Attachments)
    retrieved_memory: Optional[List[str]]
    standup_data: Optional[Dict[str, Any]]
    sprint_insights: Optional[Dict[str, Any]]
    project_insights: Optional[Dict[str, Any]]
    mcp_insights: Optional[Dict[str, Any]]
    file_id: Optional[str]
    attached_document: Optional[Dict[str, Any]]
    active_error: Optional[str]
    active_errors: Optional[List[str]]

    # 6. Active Intent, Staged Actions & Mutation Receipts
    active_intent: Optional[str]
    pending_action_payload: Optional[Dict[str, Any]]
    mutation_receipt: Optional[Dict[str, Any]]


class KayaState(MessagesState):
    """
    Primary Kaya State class extending LangGraph MessagesState.
    Included for backward compatibility and typed access across nodes.
    """

    user_id: str
    user_name: Optional[str]
    thread_id: str
    project_id: Optional[str]
    project_name: Optional[str]
    next: Union[str, List[str]]

    action_type: Optional[str]
    router_reasoning: Optional[str]

    # Isolated Sub-Agent Buffers
    _analyst_messages: Annotated[List[Any], reset_or_add]
    _analyst_tool_call_id: Optional[str]
    _db_write_messages: Annotated[List[Any], reset_or_add]
    _sprint_messages: Annotated[List[Any], reset_or_add]
    _mcp_messages: Annotated[List[Any], reset_or_add]
    _mcp_tool_call_id: Optional[str]

    # Shared Context & Cache
    retrieved_memory: Optional[List[str]]
    standup_data: Optional[Dict[str, Any]]
    sprint_insights: Optional[Dict[str, Any]]
    project_insights: Optional[Dict[str, Any]]
    mcp_insights: Optional[Dict[str, Any]]
    file_id: Optional[str]
    attached_document: Optional[Dict[str, Any]]
    active_error: Optional[str]
    active_errors: Optional[List[str]]

    # Active Intent, Staged Actions & Mutation Receipts
    active_intent: Optional[str]
    pending_action_payload: Optional[Dict[str, Any]]
    mutation_receipt: Optional[Dict[str, Any]]

