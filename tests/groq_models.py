import os
import sys
import asyncio
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
load_dotenv()


class SupervisorDecision(BaseModel):
    actions: List[str] = Field(
        description="List of sub-agent actions to trigger in parallel: 'db_write', 'analyst', 'sprint', 'direct_response'"
    )
    reasoning: str = Field(description="Reasoning for routing decision.")


CANDIDATE_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "groq/compound",
]

TEST_QUERIES = [
    "Show me all critical issues and team member workloads",
    "Create a sprint called auth-v1 and show sprint velocity",
    "What is my standup for today and set up report scheduler",
    "hi how are you today",
]


async def test_groq_model(model_name: str) -> dict:
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    print(f"\n=======================================================")
    print(f" Testing Groq Model: '{model_name}'")
    print(f"=======================================================")

    success_count = 0
    errors = []
    sample_results = []

    for query in TEST_QUERIES:
        try:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=groq_api_key,
                openai_api_base="https://api.groq.com/openai/v1",
                temperature=0.0,
                max_retries=1,
            ).with_structured_output(SupervisorDecision)

            messages = [
                SystemMessage(content="You are a supervisor router. Classify user query into actions: 'db_write', 'analyst', 'sprint', 'direct_response'."),
                HumanMessage(content=f"User Query: '{query}'"),
            ]

            decision: SupervisorDecision = await llm.ainvoke(messages)
            success_count += 1
            sample_results.append({
                "query": query,
                "actions": decision.actions,
                "reasoning": decision.reasoning,
            })
            print(f"  [SUCCESS] Query: '{query}' -> Actions: {decision.actions} | Reasoning: {decision.reasoning}")

        except Exception as e:
            error_str = str(e)
            if "model_not_found" in error_str or "404" in error_str:
                error_summary = "404 Model Not Found / No Access on Groq API"
            else:
                error_summary = error_str[:120]
            errors.append(error_summary)
            print(f"  [FAILED] Query: '{query}' -> Error: {error_summary}")

    return {
        "model": model_name,
        "available": success_count > 0,
        "success_rate": f"{success_count}/{len(TEST_QUERIES)}",
        "samples": sample_results,
        "errors": list(set(errors)),
    }


async def main():
    print("Starting Groq 4 Models Benchmark...\n")
    results = []
    for model in CANDIDATE_MODELS:
        res = await test_groq_model(model)
        results.append(res)

    print("\n\n" + "=" * 70)
    print("                  GROQ 4 MODELS BENCHMARK SUMMARY")
    print("=" * 70)
    for r in results:
        status = "AVAILABLE" if r["available"] else "UNAVAILABLE"
        print(f"\nModel: {r['model']} -> {status} ({r['success_rate']})")
        if not r["available"]:
            print(f"   Reason: {r['errors']}")
        else:
            for s in r['samples']:
                print(f"   Query: '{s['query']}'")
                print(f"     -> Actions: {s['actions']}")
                print(f"     -> Reasoning: {s['reasoning']}\n")


if __name__ == "__main__":
    asyncio.run(main())
