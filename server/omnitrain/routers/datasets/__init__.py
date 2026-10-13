"""The dataset pool: snapshot versions, items, uploads, splits, labeling, health, annotations.

A project has one dataset (1:1). Items live in a project-wide, content-addressed, deduplicated pool
(dataset_items); a version (the mutable draft, or an immutable snapshot) is just a set of pool items
with a split assignment (dataset_version_items). Snapshotting copies pool-item membership into a new
version and kicks off an async parquet build (services/snapshot.py); clients poll GET /versions/{id}
for status ready | failed.

The routes live in one module per concern; this package only assembles them, in the order they are documented.
"""

from fastapi import APIRouter

from omnitrain.routers.datasets import annotations, bulk, health, listing, upload, versions

router = APIRouter(tags=["datasets"])
for _module in (versions, listing, upload, bulk, health, annotations):
    router.include_router(_module.router)
