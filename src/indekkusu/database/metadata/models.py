from peewee import AutoField, BlobField, CharField, BigIntegerField

from .base import db


class Image(db.Model):
    id = AutoField()
    hash = BlobField(index=True)
    path = CharField()


class VectorNumber(db.Model):
    id = AutoField()
    num = BigIntegerField(index=True)
