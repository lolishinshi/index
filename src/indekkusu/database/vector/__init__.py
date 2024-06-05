from playhouse.sqliteq import SqliteQueueDatabase

from .base import db
from .models import Vector

__all__ = ["Vector", "connect"]


def connect(path: str, readonly: bool = False):
    database = SqliteQueueDatabase(
        f"file:{path}?mode=ro" if readonly else path,
        timeout=5,
        pragmas={
            "journal_mode": "wal",
            "synchronous": 1,  # normal
            "cache_size": -64 * 1000,  # 64MB page cache
        },
    )
    db.initialize(database)
    database.create_tables([Vector])
