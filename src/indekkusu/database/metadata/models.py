from peewee import AutoField, BlobField, CharField

from .base import db


class Image(db.Model):
    id = AutoField()
    hash = BlobField(index=True)
    path = CharField()
