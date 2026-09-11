from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.routes.admin import router as admin_router
from app.api.routes.auth import router as auth_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.drift_evaluation import router as drift_evaluation_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.health import router as health_router
from app.api.routes.screening import router as screening_router
from app.auth import AuthConfigurationError, AuthRepository, DatabaseUnavailable, bootstrap_admin, get_settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="BurnUI API",
        version="0.1.0",
        description="Backend service for component-screening workflows.",
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(admin_router, prefix="/api")
    app.include_router(datasets_router, prefix="/api")
    app.include_router(drift_evaluation_router, prefix="/api")
    app.include_router(evaluation_router, prefix="/api")
    app.include_router(screening_router, prefix="/api")

    @app.exception_handler(DatabaseUnavailable)
    async def database_unavailable(_: Request, __: DatabaseUnavailable) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": {"message": "The BurnUI identity database is unavailable."}})

    @app.exception_handler(AuthConfigurationError)
    async def authentication_configuration_error(_: Request, exc: AuthConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": {"message": str(exc)}})

    @app.on_event("startup")
    def initialize_identity_store() -> None:
        settings = get_settings()
        repository = AuthRepository(settings)
        repository.initialize()
        bootstrap_admin(repository, settings)

    return app


app = create_app()
