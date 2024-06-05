from collections import Counter, defaultdict
from pathlib import Path

import click
from loguru import logger

from indekkusu.database import connect, crud
from indekkusu.index.train import FaissIndexTrainer

from .base import cli, click_db_dir


@cli.command()
@click_db_dir
@click.option("--gpu", is_flag=True, help="使用 GPU 训练索引")
def build_index(db_dir: Path):
    """
    构建索引
    """
    connect(str(db_dir))

    # 第一步，遍历所有向量，找出重复的向量
    logger.info("查找重复向量中")
    d = set()
    h = set()
    for vector in crud.vector.iter_by(0, 100000000):
        for v in vector.vector:
            hash = bytes(v)
            if hash in h:
                d.add(hash)
            else:
                h.add(hash)
    del h

    # 第二步，找出重复的向量对应的图片
    logger.info("查找重复向量对应的图片中")
    g = set()
    m = defaultdict(list)
    for vector in crud.vector.iter_by(0, 100000000):
        for v in vector.vector:
            hash = bytes(v)
            if hash in d:
                m[hash].append(vector.id)
                g.add(vector.id)

    # 第三步，筛选特征点重复度高的图片
    logger.info("筛选特征点重复度高的图片中")
    counter = Counter(v for values in m.values() for v in values)
