from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict[str, str]:
    """Confirm that the API process is available."""
    return {"status": "ok", "service": "burnui-api"}
