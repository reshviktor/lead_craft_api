from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from src.config import settings


_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=_is_sqlite,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

LocalSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with LocalSessionLocal() as session:
        yield session

