import os

from playhouse.sqliteq import SqliteQueueDatabase

from .base import db
from .models import Image, IndexStatus

__all__ = ["Image", "connect"]


def connect(path: str | os.PathLike, readonly: bool = False):
    database = SqliteQueueDatabase(
        f"file:{path}?mode=ro" if readonly else str(path),
        timeout=5,
        pragmas={
            "journal_mode": "wal",
            "synchronous": 1,  # normal
        },
    )
    db.initialize(database)
    database.create_tables([Image, IndexStatus])
