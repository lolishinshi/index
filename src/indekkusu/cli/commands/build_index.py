from pathlib import Path
from typing import Generator

import numpy as np
import click
from loguru import logger

from indekkusu.database import connect, crud

from .base import cli, click_db_dir


@cli.command()
@click_db_dir
@click.option("-i", "--index-id", default=0, show_default=True, help="索引 ID")
@click.option("-n", "--num", help="限制添加的图片数量")
@click.option(
    "-c", "--chunk", default=1000, show_default=True, help="每批添加多少张图片到索引中"
)
def build_index(db_dir: Path, index_id: int, num: int | None, chunk: int):
    """
    构建索引
    """
    connect(str(db_dir))

    id_start = crud.image.get_indexed() + 1

    for chunk in chunk_index(id_start, num, chunk):
        logger.info("Adding {} vectors to index", len(chunk))
        pass


def chunk_index(
    start: int, limit: int | None, chunk_size: int
) -> Generator[np.ndarray, None, None]:
    cache = []
    for v in crud.vector.iter_by(start, limit):
        cache.append(v.vector)
        if len(cache) == chunk_size:
            yield np.concatenate(cache)
            cache.clear()
    if cache:
        yield np.concatenate(cache)
