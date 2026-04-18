# app/agent/graph.py

import os
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
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
    model=os.getenv("KAYA_MODEL", "gpt-4.1-nano"),
    temperature=0.3,
    streaming=True,
)

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


def kaya(state: KayaState) -> dict:
    print(f"[kaya] messages={len(state['messages'])}")
    messages = [SystemMessage(content=KAYA_SYSTEM)] + state["messages"]
    response = _llm.invoke(messages)
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
