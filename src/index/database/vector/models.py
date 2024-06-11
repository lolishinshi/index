import io

import numpy as np
from peewee import BlobField, IntegerField

from .base import db


class NDArrayField(BlobField):
    def db_value(self, value: np.ndarray):
        return value.tobytes()

    def python_value(self, value: bytes):
        return np.frombuffer(value, dtype=np.uint8).reshape(-1, 32)


class Vector(db.Model):
    id = IntegerField(primary_key=True)
    vector = NDArrayField()
