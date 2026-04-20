from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import agent, state, debug


def create_app() -> FastAPI:
    app = FastAPI(title="KAYA-wekraft", version="1.0.0")

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # routers
    app.include_router(agent.router)
    app.include_router(state.router)
    app.include_router(debug.router)

    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    print("[SERVER] FastAPI app initialized and routes included")
