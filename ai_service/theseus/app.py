"""FastAPI application factory.

create_app() must stay importable with no side effects (no DB, no S3, no network): the OpenAPI
export script imports it at build time to dump schema/openapi.json for Orval. Everything that
touches infrastructure lives in lifespan.py and only runs when the server actually starts.
"""

from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute

from theseus.augmentation.registry import list_augmentations
from theseus.backends.registry import list_backends
from theseus.deps import verify_origin
from theseus.errors import install_error_handlers
from theseus.export.registry import list_export_formats
from theseus.lifespan import lifespan
from theseus.routers import (
    api_keys,
    api_v1,
    auth,
    classes,
    datasets,
    export,
    health,
    inference,
    projects,
    sweeps,
    training,
)


def _operation_id(route: APIRoute) -> str:
    """Orval derives hook names from operationId, so use the (unique, stable) handler name."""
    return route.name


def create_app() -> FastAPI:
    app = FastAPI(
        title="CTU Theseus API",
        version="1.0.0",
        lifespan=lifespan,
        generate_unique_id_function=_operation_id,
        dependencies=[Depends(verify_origin)],
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    install_error_handlers(app)
    # Import every export format, augmentation and trainer backend plugin now, so a duplicate id or
    # a broken plugin module stops the backend at startup instead of failing the first request that
    # needs it. Import only: no DB, S3 or network (a backend's own heavy ML import is lazy, see
    # backends/base.py), so this stays safe for the OpenAPI export script.
    list_export_formats()
    list_augmentations()
    list_backends()

    api = APIRouter(prefix="/api")
    api.include_router(auth.router)
    api.include_router(projects.router)
    api.include_router(classes.router)
    api.include_router(datasets.router)
    api.include_router(api_keys.router)
    api.include_router(training.router)
    api.include_router(sweeps.router)
    api.include_router(inference.router)
    api.include_router(export.router)
    api.include_router(api_v1.router)
    app.include_router(api)
    app.include_router(health.router)
    return app


app = create_app()
