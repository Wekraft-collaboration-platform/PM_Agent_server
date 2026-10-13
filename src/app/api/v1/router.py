from fastapi import APIRouter
from app.api.v1.endpoints.kaya import router as kaya_router
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.harry import router as harry_router
from app.api.v1.endpoints.state import router as state_router
from app.api.v1.endpoints.debug import router as debug_router

api_router = APIRouter()

# Main active focus router (/kaya and /agent alias)
api_router.include_router(kaya_router)

# Document upload & layout-aware parsing router
api_router.include_router(documents_router)

# Also expose under /agent for backwards compatibility
agent_alias_router = APIRouter(prefix="/agent", tags=["Kaya AI Agent"])
for route in kaya_router.routes:
    agent_alias_router.routes.append(route)
api_router.include_router(agent_alias_router)

# Future work routers
api_router.include_router(harry_router)

# State & Debug routers
api_router.include_router(state_router)
api_router.include_router(debug_router)

