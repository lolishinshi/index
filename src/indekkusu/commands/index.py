import threading
from datetime import datetime
from pathlib import Path
from queue import Queue

import click
import numpy as np
from loguru import logger

from indekkusu.database import connect, crud
from indekkusu.index import FaissIndexManager, FaissIndexTrainer

from .base import cli, click_db_dir


@cli.group()
def index():
    """
    索引管理
    """
    pass


@index.command()
@click_db_dir
@click.option("-l", "--limit", help="限制添加的图片数量")
@click.option(
    "-c", "--chunk", default=50000, show_default=True, help="每批添加多少张图片到索引中"
)
@click.option("-n", "--name", help="索引名称", required=True)
@click.option("--description", help="索引描述", required=True)
@click.option(
    "-i", "--interval", default=60, show_default=True, help="保存索引的间隔，单位秒"
)
def build(
    db_dir: Path,
    limit: int | None,
    chunk: int,
    description: str,
    name: str,
    interval: int,
):
    """
    构建索引
    """
    connect(str(db_dir))
    m = FaissIndexManager(db_dir, description)
    index = m.get_index(name)

    id_start = crud.image.get_indexed(name) + 1
    logger.info("开始添加图片到索引，起始 ID: {}", id_start)

    queue = Queue(1)
    start = datetime.now()

    chunk_thread = threading.Thread(
        target=chunk_index, args=(id_start, limit, chunk, queue)
    )
    chunk_thread.start()

    while True:
        r = queue.get()
        if r is None:
            break
    
        ilist, vlist = r
        logger.info("正在增加 {} 张图片", len(vlist))
        xids = np.concatenate(ilist)
        vectors = np.concatenate(vlist)
        index.add_with_ids(vectors, xids)

        if (datetime.now() - start).seconds > interval:
            logger.info("不平衡度: {}", index.imbalance())
            logger.info("保存索引")
            index.save()
            crud.image.add_indexed(name, len(vlist))
            start = datetime.now()
            vlist = []

    if vlist:
        index.save()
        crud.image.add_indexed(name, len(vlist))


def chunk_index(start: int, limit: int | None, chunk_size: int, queue: Queue):
    xids = []
    vectors = []
    for v in crud.vector.iter_by(start, limit):
        xids.append(v.id << 10 | np.arange(v.vector.shape[0], dtype=np.uint64))
        vectors.append(v.vector)
        if len(vectors) == chunk_size:
            queue.put((xids, vectors))
            xids = []
            vectors = []
    if vectors:
        queue.put((xids, vectors))
    queue.put(None)


@index.command()
@click_db_dir
@click.option(
    "-i",
    "--image-num",
    default=10000,
    show_default=True,
    help="通过预计索引的图片数量来创建索引",
)
@click.option("-d", "--description", help="使用索引描述来创建索引")
@click.option(
    "-n", default=50, show_default=True, help="使用多少倍的特征点训练索引，推荐 30~256"
)
@click.option("--gpu", is_flag=True, help="使用 GPU 训练索引")
def train(
    db_dir: Path,
    image_num: int,
    n: int,
    gpu: bool,
    description: str | None,
):
    """
    构建索引
    """
    connect(str(db_dir))
    max_size = image_num * 500
    trainer = FaissIndexTrainer(db_dir, max_size, description)
    logger.info("创建索引 {}", trainer.description)
    vectors = crud.vector.sample(n * trainer.k // 500)
    logger.info("采样 {}/{} 向量，开始训练", len(vectors), n * trainer.k)
    if gpu:
        trainer.train_gpu(vectors)
    else:
        trainer.train(vectors)
    logger.info("训练完成")
    trainer.save()
