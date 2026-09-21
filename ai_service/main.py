"""Process entrypoint: apply database migrations, then serve the API.

    python main.py

Migrations run here, before uvicorn starts, and NOT in the FastAPI lifespan: a slow migration
inside lifespan would fail the health check, and it is the wrong place for it structurally.

The service must run as exactly ONE process (see theseus.lifespan.assert_single_process), so this
never passes workers or reload to uvicorn.
"""

import logging
import os
from pathlib import Path

import uvicorn
from alembic import command
from alembic.config import Config

from theseus.settings import get_settings

HERE = Path(__file__).resolve().parent


def run_migrations() -> None:
    cfg = Config(str(HERE / "alembic.ini"))
    cfg.set_main_option("script_location", str(HERE / "migrations"))
    command.upgrade(cfg, "head")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    settings = get_settings()
    if not os.environ.get("SKIP_MIGRATIONS"):
        run_migrations()
    from theseus.app import app

    uvicorn.run(app, host="0.0.0.0", port=settings.port, log_config=None)


if __name__ == "__main__":
    main()
