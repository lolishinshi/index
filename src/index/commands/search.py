from datetime import datetime
from pathlib import Path

import click
from loguru import logger

from index.database import connect, crud
from index.feature import FeatureExtractor
from index.index import FaissIndexManager
from index.utils import load_image

from .base import cli, click_db_dir


@cli.command()
@click_db_dir
@click.option("-l", "--limit", default=10, show_default=True, help="返回结果数量")
@click.option("-n", "--name", required=True, help="索引文件名称")
@click.option("--mmap", is_flag=True, help="使用 mmap 加载索引")
@click.argument("image", type=click.Path(exists=True))
def search(db_dir: Path, image: str, name: str, limit: int, mmap: bool):
    """
    搜索图片
    """
    connect(str(db_dir))

    m = FaissIndexManager(db_dir)
    index = m.get_index(name, mmap)

    img = load_image(image)
    _, desc = FeatureExtractor().detect_and_compute(img)

    result = index.search(desc, limit)
    for k, v in result.__dict__.items():
        if k != "result":
            logger.debug("{}: {}", k, v)

    now = datetime.now()
    images = [(score, crud.image.get_by_id(id_).path) for id_, score in result.result]
    logger.debug("db_time: {}", (datetime.now() - now).total_seconds() * 100)

    for score, path in images:
        logger.info("{:.2f} | {}", score, path)
