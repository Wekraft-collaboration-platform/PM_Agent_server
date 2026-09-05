from fastapi import APIRouter, HTTPException
from app.core.utils import format_state_snapshot

router = APIRouter(tags=["state"])


@router.get("/state")
async def state(thread_id: str | None = None):
    """Endpoint returning current graph state snapshot."""
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    return {"thread_id": thread_id, "status": "active"}


@router.get("/history")
async def history(thread_id: str | None = None):
    """Endpoint returning state history records."""
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required")
    return []
