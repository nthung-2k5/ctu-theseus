"""Uniform error envelope: {"error": {"code": ..., "message": ...}} (same shape the Elysia gateway used)."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    422: "VALIDATION",
    429: "RATE_LIMITED",
    503: "UNAVAILABLE",
}


def envelope(status: int, message: str, code: str | None = None) -> dict:
    return {"error": {"code": code or _CODES.get(status, "ERROR"), "message": message}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        headers = getattr(exc, "headers", None)
        return JSONResponse(envelope(exc.status_code, str(exc.detail)), exc.status_code, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = f"{where}: {first.get('msg')}" if where else str(first.get("msg", "Invalid request"))
        return JSONResponse(envelope(422, message), 422)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(envelope(500, "Internal Server Error", "INTERNAL"), 500)


__all__ = ["HTTPException", "envelope", "install_error_handlers"]
