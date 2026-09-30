"""/api/admin: everything an administrator can manage.

The whole router sits behind `require_admin`, so a non-admin gets 403 on every path here, including
ones a later phase adds. Handler names are the OpenAPI operationIds (see app._operation_id), so each
one carries an `admin_` prefix to stay unique across routers.
"""

from fastapi import APIRouter, Depends

from theseus.deps import require_admin
from theseus.routers.admin import models, plugins, users

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])
router.include_router(users.router)
router.include_router(plugins.router)
router.include_router(models.router)
