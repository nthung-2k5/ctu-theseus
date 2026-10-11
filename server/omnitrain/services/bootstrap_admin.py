"""Make sure the instance has an administrator from its very first start.

Without this, the first admin only exists after someone registers an account whose email is listed in
OMNITRAIN_ADMIN_EMAILS. Here startup seeds one instead, but only when the database has no admin at all: an
existing admin is never touched, and deleting or demoting the seeded account later does not bring it back as
long as some other admin exists.
"""

import logging

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from omnitrain.auth.password import hash_password
from omnitrain.db.base import get_sessionmaker
from omnitrain.db.models import User
from omnitrain.settings import get_settings

logger = logging.getLogger(__name__)


async def ensure_admin() -> None:
    s = get_settings()
    password = s.admin_password
    if not password:
        if s.is_production:
            logger.warning(
                "No admin account was seeded: set OMNITRAIN_ADMIN_PASSWORD (and OMNITRAIN_ADMIN_EMAIL) in production."
            )
            return
        password = s.dev_admin_password
    email = s.admin_email.strip().lower()

    async with get_sessionmaker()() as session:
        has_admin = (await session.execute(sa.select(User.id).where(User.role == "admin").limit(1))).first()
        if has_admin:
            return
        existing = (await session.execute(sa.select(User).where(User.email == email))).scalar_one_or_none()
        if existing is not None:
            # The address is taken by an ordinary account: promote it rather than invent a second one. Its
            # password stays as the owner set it.
            existing.role = "admin"
            existing.disabled_at = None
            await session.commit()
            logger.info("Promoted existing account %s to admin (no admin existed).", email)
            return
        session.add(User(name="Administrator", email=email, password_hash=await hash_password(password), role="admin"))
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()  # lost a race with a concurrent registration of the same email
            return

    if s.admin_password:
        logger.info("Created the admin account %s.", email)
    else:
        logger.warning(
            "Created the admin account %s with the development password %r. Set OMNITRAIN_ADMIN_PASSWORD to choose "
            "your own.",
            email,
            password,
        )
