import os
import sys
import asyncio
import uuid
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src/ to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
load_dotenv(override=True)

from langchain_core.messages import HumanMessage
from app.core.cache import run_guardrails_and_cache_parallel
from app.agents.graph.kaya_graph import kaya_graph


TEST_USER_ID = "nd7088ge1w6p1nb20spjzwnsw98dtf3z"
TEST_PROJECT_ID = "kn71qtgem0xhc05gwq2yxt2cxx8dv1w9"

QUERIES = [
    ("Query 1", "im managing project , but i have a headache please suggest 5 medicines that makes me feel again active."),
    ("Query 2", "what are the tasks and issues in this project ?"),
    ("Query 3", "hey , what u can do ?"),
    ("Query 4", "what the workload of our team ?"),
]


async def run_single_test(label: str, query_text: str, user_id: str, project_id: str):
    print("=" * 80)
    print(f"TEST: {label}")
    print(f"INPUT QUERY: {query_text}")
    print(f"USER ID: {user_id}")
    print(f"PROJECT ID: {project_id}")
    print("-" * 80)

    # 1. GUARDRAIL & CACHE STAGE
    print("\n[STAGE 1: INPUT GUARDRAILS & CACHE]")
    pipeline_res = await run_guardrails_and_cache_parallel(query_text)
    guardrail = pipeline_res.get("guardrail", {})
    cache_hit = pipeline_res.get("cache_hit", False)

    print(f"Guardrail Safe: {guardrail.get('is_safe')}")
    print(f"Guardrail Risk Score: {guardrail.get('risk_score')}")
    print(f"Guardrail Reason: {guardrail.get('reason')}")
    print(f"Semantic Cache Hit: {cache_hit}")

    # If blocked by guardrail, stop here as in production
    if not guardrail.get("is_safe"):
        print("\n[FINAL OUTCOME: BLOCKED BY GUARDRAIL]")
        print(f"Block Reason: {guardrail.get('reason')}")
        return {
            "query": query_text,
            "status": "BLOCKED",
            "guardrail": guardrail,
            "final_response": f"Request blocked by safety guardrails: {guardrail.get('reason')}",
        }

    # If cache hit, return cached response
    if cache_hit:
        cached_response = pipeline_res.get("cached_response")
        print("\n[FINAL OUTCOME: CACHE HIT]")
        print(f"Cached Response: {cached_response}")
        return {
            "query": query_text,
            "status": "CACHE_HIT",
            "final_response": cached_response,
        }

    # 2. LANGGRAPH EXECUTION STAGE
    print("\n[STAGE 2: LANGGRAPH MULTI-AGENT EXECUTION]")
    thread_id = f"test-thread-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id, "model": "fast"}}

    initial_state = {
        "messages": [HumanMessage(content=query_text)],
        "user_id": user_id,
        "user_name": "Test User",
        "project_id": project_id,
        "thread_id": thread_id,
    }

    try:
        final_state = await kaya_graph.ainvoke(initial_state, config=config)

        print(f"Supervisor Next Node(s): {final_state.get('next')}")
        print(f"Supervisor Reasoning: {final_state.get('router_reasoning')}")
        print(f"Action Type: {final_state.get('action_type')}")

        # Sub-agent findings if executed
        if final_state.get("_analyst_messages"):
            print("\n[ANALYST NODE FINDINGS]:")
            for m in final_state.get("_analyst_messages", []):
                if isinstance(m, dict) and m.get("role") == "analyst":
                    print(m.get("content"))

        if final_state.get("_db_write_messages"):
            print("\n[DB WRITE NODE FINDINGS]:")
            for m in final_state.get("_db_write_messages", []):
                if isinstance(m, dict) and m.get("role") == "db_write":
                    print(m.get("content"))

        if final_state.get("_sprint_messages"):
            print("\n[SPRINT NODE FINDINGS]:")
            for m in final_state.get("_sprint_messages", []):
                if isinstance(m, dict) and m.get("role") == "sprint":
                    print(m.get("content"))

        # Final Response from Kaya
        messages = final_state.get("messages", [])
        last_message = messages[-1] if messages else None
        final_text = last_message.content if last_message else "No response generated"

        print("\n[FINAL OUTCOME: KAYA SYNTHESIZED RESPONSE]")
        print(final_text)

        return {
            "query": query_text,
            "status": "COMPLETED",
            "router_decision": final_state.get("next"),
            "final_response": final_text,
        }

    except Exception as e:
        import traceback
        print(f"\n[ERROR DURING GRAPH EXECUTION]: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "query": query_text,
            "status": "ERROR",
            "error": f"{type(e).__name__}: {e}",
        }


async def main():
    print("=" * 80)
    print("STARTING REAL END-TO-END PIPELINE TESTS")
    print("=" * 80)

    results = []
    for label, query in QUERIES:
        res = await run_single_test(label, query, TEST_USER_ID, TEST_PROJECT_ID)
        results.append(res)
        print("\n" + "=" * 80 + "\n")

    print("\n--- SUMMARY OF ALL 4 TEST RUNS ---")
    for idx, r in enumerate(results, start=1):
        print(f"{idx}. Status: {r.get('status')} | Query: {r.get('query')}")


if __name__ == "__main__":
    asyncio.run(main())
