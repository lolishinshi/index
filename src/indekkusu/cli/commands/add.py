import itertools
import queue
from multiprocessing import Process, Queue
from pathlib import Path
from threading import Thread
from typing import Iterator

import blake3
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

class InputImage:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        self.des = None
        self._hash = None

    @property
    def hash(self):
        if self._hash is None:
            self._hash = blake3.blake3(self.data).digest()
        return self._hash


def calc_process(input: Queue, output: Queue):
    ft = FeatureExtractor()

    while True:
        try:
            img: InputImage | None = input.get(timeout=1)
            if img is None:
                break

            arr = load_image(img.data)
            if arr is None:
                logger.warning("无法读取图片 {}", img.path)
                continue

            _, des = ft.detect_and_compute(arr)
            if len(des) == 0:
                logger.warning("无法提取特征点 {}", img.path)
                continue
            if len(des) < 500:
                logger.warning("特征点数量过少 {}", img.path)
                continue

            img.des = des

            output.put(img)
        except queue.Empty:
            continue

    output.put(None)


def feed_thread(input: Queue, images: Iterator[Path], threads: int):
    for image in images:
        img = InputImage(image)
        if crud.image.check_hash(img.hash):
            continue
        input.put(img)
    for _ in range(threads):
        input.put(None)


def write_thread(output: Queue, threads: int, total: int):
    exit_threads = 0
    with tqdm(total=total) as bar:
        while exit_threads != threads:
            img: InputImage | None = output.get()
            if img is None:
                exit_threads += 1
            else:
                key = crud.image.create(img.hash, str(img.path))
                crud.vector.create(key, img.des)
                crud.image.add_vector_num(key, len(img.des))
                bar.update()
