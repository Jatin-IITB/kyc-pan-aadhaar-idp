from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

_DEFAULT_URL = "postgresql+asyncpg://kyc:kyc@localhost:5432/kyc_idp"


class Base(DeclarativeBase):
    pass


def _get_url() -> str:
    return os.getenv("DATABASE_URL", _DEFAULT_URL)


def build_engine(url: str | None = None) -> AsyncEngine:
    return create_async_engine(
        url or _get_url(),
        pool_size=10,
        max_overflow=20,
        echo=False,
    )


def build_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


engine = build_engine()
async_session_factory = build_session_factory(engine)
