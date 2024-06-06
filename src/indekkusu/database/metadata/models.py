from peewee import AutoField, BigIntegerField, BlobField, CharField

from .base import db


class Image(db.Model):
    id = AutoField()
    hash = BlobField(index=True)
    path = CharField()


class IndexStatus(db.Model):
    indexed = BigIntegerField(default=0)
