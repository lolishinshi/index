from playhouse.sqliteq import SqliteQueueDatabase
from .base import db
from .models import Vector


__all__ = ["Vector", "connect"]


def connect(path: str):
    database = SqliteQueueDatabase(
        path,
        timeout=5,
        pragmas={
            "journal_mode": "wal",
            "synchronous": 1,  # normal
        },
    )
    db.initialize(database)
