import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add src/ to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
load_dotenv()

from app.core.cache import run_guardrails_and_cache_parallel
from app.agents.graph import route_user_request


async def process_early_input_pipeline(user_query: str, is_doc_upload: bool = False):
    print(f"\n======================================================================")
    print(f"📥 Processing Input: \"{user_query}\"")
    print(f"======================================================================")

    # 1. Parallel Guardrails & Semantic Cache Execution
    pipeline_res = await run_guardrails_and_cache_parallel(user_query, is_doc_upload=is_doc_upload)
    guardrail = pipeline_res["guardrail"]

    # Check 1: Blocked by Guardrails
    if not guardrail["is_safe"]:
        print(f"❌ BLOCKED BY GUARDRAILS: {guardrail['reason']}")
        return {
            "status": "blocked",
            "reason": guardrail["reason"],
            "router_decision": None
        }

    # Check 2: Semantic Cache Hit
    if pipeline_res["cache_hit"]:
        print(f"⚡ SEMANTIC CACHE HIT (Score: {pipeline_res['cache_score']:.4f})")
        print(f"   Cached Answer: {pipeline_res['cached_response']}")
        return {
            "status": "cache_hit",
            "cached_response": pipeline_res["cached_response"],
            "router_decision": None
        }

    # Check 3: Proceed to Supervisor Router (Groq LLM openai/gpt-oss-120b)
    print("➡️ Proceeding to Supervisor Router (Groq LLM)...")
    decision = await route_user_request(user_query)
    return {
        "status": "routed",
        "router_decision": decision
    }


async def main():
    print("🚀 Running Early Input Pipeline Benchmark (Guardrails + Cache + Router)...")

    # Sample 1: Greeting query -> Bypasses cache, passes guardrails, routes directly to Router
    print("\n--- SAMPLE 1: Greeting Query ---")
    await process_early_input_pipeline("hello")

    # Sample 2: Off-topic / Non-project code generation request -> Evaluated by Guardrails
    print("\n--- SAMPLE 2: Off-Topic / Non-Project Query ---")
    await process_early_input_pipeline("can u please give me a rust code to calculate todays time .")

    # Sample 3: User Standup & Work Items Query -> Passes guardrails, routes to Analyst ('analyst')
    print("\n--- SAMPLE 3: User Standup & Priority Query ---")
    await process_early_input_pipeline("hey , what ar emy todays work to perform ?")


if __name__ == "__main__":
    asyncio.run(main())
