from fastapi import APIRouter, HTTPException
from app.agent.graph import graph
from app.core.utils import format_state_snapshot

router = APIRouter(tags=["state"])

@router.get("/state")
async def state(thread_id: str | None = None):
    """Endpoint returning current graph state."""
    print(f"[/state] Called with thread_id={thread_id}")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    config = {"configurable": {"thread_id": thread_id}}

    state = await graph.aget_state(config)
    print(
        f"[/state] Got state for thread_id={thread_id}: next={getattr(state, 'next', None)}"
    )
    return format_state_snapshot(state)

@router.get("/history")
async def history(thread_id: str | None = None):
    """Endpoint returning complete state history. Used for restoring graph."""
    print(f"[/history] Called with thread_id={thread_id}")
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")

    config = {"configurable": {"thread_id": thread_id}}

    records = []
    async for state in graph.aget_state_history(config):
        records.append(format_state_snapshot(state))
    print(f"[/history] Returning {len(records)} records for thread_id={thread_id}")
    return records
