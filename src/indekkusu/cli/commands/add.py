import multiprocessing
import itertools
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm
from indekkusu.database import IndexkusuDB
from indekkusu.feature import FeatureExtractor
from .base import cli, click_db_dir
from ..utils import load_image


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
    计算并存储一张图片或者递归一个文件夹中的所有图片的特征点
    """
    db = IndexkusuDB(db_dir)

    if path.is_file():
        images = iter([path])
    else:
        images = itertools.chain.from_iterable(path.rglob(g) for g in glob)

    with multiprocessing.Pool(threads) as pool:
        results = pool.imap_unordered(detect_and_compute, images, chunksize=1024)
        for image, desc in tqdm(results):
            if len(desc) != 0:
                db.add_image(str(image), desc)


ft = FeatureExtractor()


def detect_and_compute(image: Path) -> tuple[Path, np.ndarray]:
    img = load_image(image)
    if img is None:
        return image, np.array([])
    _, des = ft.detect_and_compute(img)
    return image, des
