import itertools
import queue
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Thread
from typing import Iterator

import click
import numpy as np
from loguru import logger
from tqdm import tqdm

from indekkusu.database import connect, crud
from indekkusu.feature import FeatureExtractor

from ..utils import load_image
from .base import cli, click_db_dir


@cli.command()
@click_db_dir
@click.option("-t", "--threads", default=1, show_default=True, help="并发线程数")
@click.option(
    "-g",
    "--glob",
    multiple=True,
    default=["*.jpg", "*.jpeg", "*.png"],
    help="匹配文件名的 glob 表达式",
)
@click.argument("PATH", type=click.Path(exists=True, path_type=Path))
def add(db_dir: Path, path: Path, glob: list[str], threads: int):
    """
    计算并存储一个文件夹中的所有图片的特征点
    """
    connect(str(db_dir))

    total = sum(sum(1 for _ in path.rglob(g)) for g in glob)
    images = itertools.chain.from_iterable(path.rglob(g) for g in glob)

    input = Queue(maxsize=threads * 2)
    output = Queue()

    for _ in range(threads):
        Process(target=calc_process, args=(input, output)).start()

    t1 = Thread(target=feed_thread, args=(input, images, threads))
    t2 = Thread(target=write_thread, args=(output, threads, total))
    t1.start()
    t2.start()

    t2.join()


def calc_process(input: Queue, output: Queue):
    ft = FeatureExtractor()

    while True:
        try:
            key, image = input.get(timeout=1)
            if key == -1:
                break

            img = load_image(image)
            if img is None:
                logger.warning("无法读取图片 {}", image)
                continue

            _, des = ft.detect_and_compute(img)
            if len(des) == 0:
                logger.warning("无法提取特征点 {}", image)
                continue

            output.put((key, des))
        except queue.Empty:
            continue

    output.put((-1, np.array([])))


def feed_thread(input: Queue, images: Iterator[Path], threads: int):
    for image in images:
        if key := crud.image.create(str(image)):
            input.put((key, image))
    for _ in range(threads):
        input.put((-1, Path()))


def write_thread(output: Queue, threads: int, total: int):
    exit_threads = 0
    with tqdm(total=total) as bar:
        while exit_threads != threads:
            key, des = output.get()
            if key == -1:
                exit_threads += 1
            else:
                crud.vector.create(key, des)
                bar.update()
