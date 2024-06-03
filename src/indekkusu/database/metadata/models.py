from peewee import AutoField, CharField, BlobField
from .base import db


class Image(db.Model):
    id = AutoField()
    hash = BlobField(index=True)
    path = CharField()
