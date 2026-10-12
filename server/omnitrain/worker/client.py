"""The worker's view of /api/worker. Synchronous: the worker trains on its main thread and flushes from one more."""

from typing import Any

import httpx


class ApiError(Exception):
    """The API answered, and refused. `status` 5xx and 429 are worth retrying; the rest are not."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.status >= 500 or self.status == 429


class Unauthorized(ApiError):
    """The token is wrong or was revoked. Nothing the worker does will fix that."""


# A request that cannot reach the API at all, or got a retryable answer.
TRANSIENT = (httpx.TransportError, ApiError)


def is_transient(exc: BaseException) -> bool:
    return isinstance(exc, httpx.TransportError) or (isinstance(exc, ApiError) and exc.retryable)


def _message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])
    except Exception:
        return response.text[:300] or response.reason_phrase


class ApiClient:
    def __init__(
        self, base_url: str, token: str, *, timeout: float = 30.0, transport: httpx.BaseTransport | None = None
    ):
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        response = self._http.post(f"/api/worker{path}", json=body)
        if response.status_code == 401:
            raise Unauthorized(401, _message(response))
        if response.status_code >= 400:
            raise ApiError(response.status_code, _message(response))
        return response

    def claim(self, describe: dict[str, Any]) -> dict[str, Any] | None:
        """The next run for this worker (camelCase, as the API sends it), or None when nothing is waiting."""
        response = self._post("/claim", describe)
        return None if response.status_code == 204 else response.json()

    def events(self, run_id: str, metrics: list[dict[str, Any]], heartbeat: bool) -> dict[str, Any]:
        return self._post(f"/runs/{run_id}/events", {"metrics": metrics, "heartbeat": heartbeat}).json()

    def complete(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/runs/{run_id}/complete", body).json()
