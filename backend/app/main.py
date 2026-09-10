from fastapi import FastAPI

from app.api.routes.datasets import router as datasets_router
from app.api.routes.health import router as health_router
from app.api.routes.screening import router as screening_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="BurnUI API",
        version="0.1.0",
        description="Backend service for component-screening workflows.",
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(datasets_router, prefix="/api")
    app.include_router(screening_router, prefix="/api")
    return app


app = create_app()
