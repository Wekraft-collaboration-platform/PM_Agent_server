from fastapi import APIRouter

router = APIRouter(tags=["debug"])


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "agent": "kaya_v1"}
