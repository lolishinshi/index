import random
from pathlib import Path

import cv2
import click
from cv2.typing import MatLike


@click.command()
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.argument("dest", type=click.Path(path_type=Path))
@click.option("-s", "--scale", default=1.0, show_default=True, help="缩放比例")
@click.option("-m", default=3, show_default=True, help="分割行数")
@click.option("-n", default=3, show_default=True, help="分割列数")
@click.option("-c", "--count", default=100, show_default=True, help="生成数量")
def gen_search(source: Path, dest: Path, scale: float, m: int, n: int, count: int):
    """
    生成搜索测试数据集
    """
    dest = dest / f"{m}x{n}_{scale}"
    if not dest.exists():
        dest.mkdir(parents=True)

    source_images = list(source.glob("*.*"))
    random.shuffle(source_images)

    for image in source_images[:count]:
        img = cv2.imread(str(image))
        if img is None:
            continue
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        parts = split_image(img, m, n)
        for i, part in enumerate(parts):
            cv2.imwrite(str(dest / f"{image.stem}_{i}{image.suffix}"), part)


def split_image(image: MatLike, m: int, n: int) -> list[MatLike]:
    """
    将图片分割为 m * n 份
    """
    h, w = image.shape[:2]
    h_step = h // m
    w_step = w // n
    images = []
    for i in range(m):
        for j in range(n):
            img = image[i * h_step : (i + 1) * h_step, j * w_step : (j + 1) * w_step]
            images.append(img)
    return images


if __name__ == "__main__":
    gen_search()
