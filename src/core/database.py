from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker


def create_session_factory(database_url: str) -> tuple[Engine, sessionmaker[Session]]:
    connect_args: dict[str, object] = {}
    url = make_url(database_url)
    if url.drivername.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        db_path = url.database
        if db_path and db_path != ":memory:":
            db_file = Path(db_path).expanduser()
            db_file.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        database_url,
        future=True,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
    return engine, session_factory
