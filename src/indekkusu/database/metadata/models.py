from peewee import AutoField, BlobField, CharField, BigIntegerField

from .base import db


class Image(db.Model):
    id = AutoField()
    hash = BlobField(index=True)
    path = CharField()


class TotalVector(db.Model):
    id = AutoField()
    total = BigIntegerField(index=True)
