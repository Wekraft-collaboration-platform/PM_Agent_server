# Execution Tasks Checklist - WEKRAFT Production Agent Architecture

This checklist details the step-by-step tasks to implement the Semantic Cache, Router/Supervisor, and Multi-Agent Graph architecture.

---

## 📋 Task List

- [x] **Task 1: Semantic Cache Service with Cohere Embeddings & Parallel Guardrail Execution**
  - Add `COHERE_API_KEY` to `agent/.env`.
  - Build `SemanticCacheService` using Cohere API embeddings for query similarity matching.
  - Update input pipeline to run Guardrails & Semantic Cache in parallel using `asyncio.gather`.
  - Skip cache lookup if query is a greeting or file upload.


- [ ] **Task 2: Toolset Reorganization (`DBwrite_tools`, `analyst_tools`, `sprint_tools`)**
  - Rename `KAYA_TOOLS` -> `DBWRITE_TOOLS` (`search_user_memory`, `create_calendar_event`, `get_user_standup`, `setup_report_scheduler`).
  - Keep `ANALYST_TOOLS` (`get_tasks_summary`, `get_issues_summary`, `get_member_workload`, `get_project_insights`, `bulk_create_tasks`, `bulk_create_issues`).
  - Keep `SPRINT_TOOLS` (`get_sprint_insights`, `create_sprint`, `add_items_to_sprint`).

- [ ] **Task 3: Router & Pydantic `SupervisorDecision` Schema**
  - Implement `SupervisorDecision(BaseModel)` with `actions: List[str]` and `reasoning: str`.
  - Configure Groq router model (`gpt-120b`).
  - Build supervisor routing node that decides which sub-agents (`db_write_agent`, `analyst_agent`, `sprint_agent`, or direct `kaya_user_facing_agent`) to execute.

- [ ] **Task 4: Multi-Agent Graph Skeleton (`agent/src/app/agents/graph.py`)**
  - Build `db_write_agent` node.
  - Build `analyst_agent` node.
  - Build `sprint_agent` node.
  - Build `kaya_user_facing_agent` node (primary streaming response node that receives all sub-agent findings and synthesizes the user-facing response).
  - Connect graph edges so sub-agents report back to `kaya_user_facing_agent`.

- [ ] **Task 5: End-to-End Test & Verification**
  - Test parallel guardrail + cache execution.
  - Test supervisor routing decision logic using `uv run python`.
