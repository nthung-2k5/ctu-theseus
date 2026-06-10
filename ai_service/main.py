"""
Theseus AI Microservice -- Minimal health endpoint (optional).

The AI service is now a NATS consumer (see worker.py).
This module only exists to provide a /health endpoint for Docker probes.
The main entry point is `python -m ai_service.worker`.

If you don't need a health endpoint, this file can be removed.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CTU Theseus AI Service",
    description="Health check endpoint. All task processing happens via NATS.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint for Docker / load-balancer probes."""
    return {"status": "ok", "transport": "nats"}
