from typing import Generator

import numpy as np
from peewee import fn

from ..vector import Vector

__all__ = ["create", "iter_by"]


def create(key: int, vector: np.ndarray) -> None:
    """
    创建一个向量记录
    """
    Vector.create(id=key, vector=vector)


def iter_by(start: int, limit: int | None = None) -> Generator[Vector, None, None]:
    """
    遍历 [start, end) 的向量记录
    """
    return (
        Vector.select()
        .where(Vector.id >= start)
        .order_by(Vector.id)
        .limit(limit)
        .iterator()
    )


def sample(n: int) -> np.ndarray:
    """
    随机采样 n 个图片的向量
    """
    arr = np.concatenate(
        [v.vector for v in Vector.select().order_by(fn.Random()).limit(n)]
    )
    return arr
