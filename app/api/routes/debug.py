from fastapi import APIRouter

router = APIRouter(tags=["debug"])


@router.get("/health")
async def health():
    print("[/health] Health check called")
    return {"status": "ok", "agent": "kaya_v1"}
