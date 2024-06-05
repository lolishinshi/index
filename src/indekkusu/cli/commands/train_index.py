from pathlib import Path

import click
from loguru import logger

from indekkusu.database import connect, crud
from indekkusu.index.train import FaissIndexTrainer

from .base import cli, click_db_dir


@cli.command()
@click_db_dir
@click.option(
    "-i", "--image-num", default=10000, show_default=True, help="预计索引的图片数量"
)
@click.option(
    "-f", "--feature-num", default=500, show_default=True, help="每张图片特征点数量"
)
@click.option(
    "-n", default=50, show_default=True, help="使用多少倍的特征点训练索引，推荐 30~256"
)
@click.option("--gpu", is_flag=True, help="使用 GPU 训练索引")
def train_index(db_dir: Path, image_num: int, feature_num: int, n: int, gpu: bool):
    """
    构建索引
    """
    connect(str(db_dir))
    max_size = image_num * feature_num
    trainer = FaissIndexTrainer(db_dir, max_size)
    logger.info("创建索引 {}", trainer.description)
    vectors = crud.vector.sample(n * trainer.k // feature_num)
    logger.info("采样 {}/{} 向量，开始训练", len(vectors), n * trainer.k)
    if gpu:
        trainer.train_gpu(vectors)
    else:
        trainer.train(vectors)
    logger.info("训练完成")
    trainer.save()
