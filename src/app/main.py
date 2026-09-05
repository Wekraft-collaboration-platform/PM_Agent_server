from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router


def create_app() -> FastAPI:
    """FastAPI Application Factory for WEKRAFT Agents Platform."""
    app = FastAPI(
        title="wekraft Agents Platform",
        version="1.0.0",
        description="Production Agent Platform supporting Kaya (active focus), Harry, and Project-Brain agents.",
    )

    # Middleware Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Master API Router (/kaya, /harry, /project-brain, /state, /health)
    app.include_router(api_router)

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    print("[SERVER] FastAPI Agent App initialized")
    print("[SERVER] Active Route: /kaya")
    print("[SERVER] Future Routes: /harry, /project-brain")
