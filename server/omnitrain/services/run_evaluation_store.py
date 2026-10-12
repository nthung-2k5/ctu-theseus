"""Recording a run's evaluation row. Shared by the local train lane (jobs/train.py) and the route a remote worker
reports through (routers/worker.py), since the evaluation itself is built by `jobs.run_core.evaluate` wherever the run
trained and only this part needs the database."""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert

from omnitrain.db.base import get_sessionmaker
from omnitrain.db.models import RunEvaluation

logger = logging.getLogger("omnitrain.jobs.train")


async def save_evaluation(run_id: uuid.UUID, **values: Any) -> None:
    """Upsert the run's single evaluation row (run_id is its primary key)."""
    values = {**values, "evaluated_at": datetime.now(UTC)}
    stmt = pg_insert(RunEvaluation).values(run_id=run_id, **values)
    async with get_sessionmaker()() as s:
        await s.execute(stmt.on_conflict_do_update(index_elements=["run_id"], set_=values))
        await s.commit()


async def record_evaluation(run_id: uuid.UUID, outcome: dict[str, Any]) -> None:
    """Store what `run_core.evaluate` produced. Like the evaluation itself, never fails the run."""
    try:
        await save_evaluation(run_id, **outcome)
    except Exception as e:
        logger.exception("Could not record the evaluation for run %s (the training result is unaffected)", run_id)
        if outcome.get("status") != "success":
            return
        try:
            await save_evaluation(run_id, status="failed", failed_message=str(e)[:2000])
        except Exception:
            logger.exception("Could not record the evaluation failure for run %s", run_id)
