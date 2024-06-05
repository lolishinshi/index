from playhouse.sqliteq import SqliteQueueDatabase

from .base import db
from .models import Image, IndexStatus, VectorNumber

__all__ = ["Image", "connect"]


def connect(path: str, readonly: bool = False):
    database = SqliteQueueDatabase(
        f"file:{path}?mode=ro" if readonly else path,
        timeout=5,
        pragmas={
            "journal_mode": "wal",
            "synchronous": 1,  # normal
        },
    )
    db.initialize(database)
    database.create_tables([Image, VectorNumber, IndexStatus])
