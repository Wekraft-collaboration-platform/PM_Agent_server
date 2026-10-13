from fastapi import APIRouter

router = APIRouter(prefix="/harry", tags=["Harry Agent (Future Work)"])


@router.get("")
@router.get("/")
async def harry_info():
    """Future Harry Agent Endpoint Placeholder."""
    return {
        "status": "planned",
        "agent": "Harry",
        "message": "Harry Agent router registered for future expansion."
    }
