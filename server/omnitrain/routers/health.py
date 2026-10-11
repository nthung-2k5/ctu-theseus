from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    """Liveness probe. Aspire waits on this before starting dependents, so it must not touch the DB."""
    return {"status": "healthy"}
