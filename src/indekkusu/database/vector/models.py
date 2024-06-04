import io

import numpy as np
from peewee import BlobField, IntegerField

from .base import db


class NDArrayField(BlobField):
    def db_value(self, value):
        with io.BytesIO() as f:
            np.save(f, value, allow_pickle=False)
            return f.getvalue()

    def python_value(self, value):
        return np.load(io.BytesIO(value), allow_pickle=False)


class Vector(db.Model):
    id = IntegerField(primary_key=True)
    vector = NDArrayField()
