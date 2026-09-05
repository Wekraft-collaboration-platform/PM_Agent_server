from fastapi import APIRouter

router = APIRouter(prefix="/project-brain", tags=["Project Brain (Future Work)"])


@router.get("")
@router.get("/")
async def project_brain_info():
    """Future Project Brain Endpoint Placeholder."""
    return {
        "status": "planned",
        "agent": "Project-Brain",
        "message": "Project Brain router registered for future expansion."
    }
