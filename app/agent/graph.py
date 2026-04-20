# app/agent/graph.py

import os
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import (
    InMemorySaver,
)  # in memory (per thread / in-session)
from mem0 import MemoryClient  # long-term memory (per user) crfoss session
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────


class KayaState(MessagesState):
    user_id: str
    thread_id: str


# ─────────────────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────────────────

_llm = ChatOpenAI(
    model=os.getenv("KAYA_MODEL", "gpt-4.1-mini"),
    temperature=0.3,
    streaming=True,
)

_mem0 = MemoryClient()
# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

KAYA_SYSTEM = """You are Kaya, an AI Product Manager agent.
You help product teams clarify requirements, write PRDs, break down features into tasks,
prioritize backlogs, and think through product decisions with clarity and structure.

Be concise, opinionated, and practical. Ask clarifying questions when needed.
Always think from the user's perspective and business impact."""

# ─────────────────────────────────────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────────────────────────────────────


# def kaya(state: KayaState) -> dict:
#     print(f"[kaya] messages={len(state['messages'])}")
#     messages = [SystemMessage(content=KAYA_SYSTEM)] + state["messages"]
#     response = _llm.invoke(messages)
#     return {"messages": [response]}


def kaya(state: KayaState) -> dict:
    user_id = state["user_id"]
    messages = state["messages"]

    print(f"[kaya] user={user_id} messages={len(messages)}")

    # ── 1. Recall: fetch relevant long-term memories for this user ──
    last_user_msg = messages[-1].content
    recalled = _mem0.search(last_user_msg, filters={"user_id": user_id})

    memory_block = ""
    results = recalled.get("results", [])
    if results:
        memory_block = "\n\nRelevant context from past sessions:\n"
        for m in results:
            memory_block += f"- {m['memory']}\n"

    # ── 2. Build prompt with injected memories ──
    system_content = KAYA_SYSTEM + memory_block
    full_messages = [SystemMessage(content=system_content)] + messages
    response = _llm.invoke(full_messages)

    # ── 3. Store: save this exchange as a new memory ──
    try:
        _mem0.add(
            [
                {"role": "user", "content": last_user_msg},
                {"role": "assistant", "content": response.content},
            ],
            user_id=user_id,
        )
        print(f"[mem0] memory saved for user={user_id}")
    except Exception as e:
        print(f"[mem0] failed to save memory: {e}")

    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────────────────────

checkpointer = InMemorySaver()


def build_graph():
    g = StateGraph(KayaState)
    g.add_node("kaya", kaya)
    g.add_edge(START, "kaya")
    g.add_edge("kaya", END)
    return g.compile(checkpointer=checkpointer)


graph = build_graph()
print("[graph] Kaya agent graph compiled successfully")
