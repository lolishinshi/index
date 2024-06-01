import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import click
from indekkusu.feature import FeatureExtractor
from indekkusu.database import IndexkusuDB
from indekkusu.utils import resize_image
from tqdm import tqdm
from loguru import logger


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-d",
    "--db-dir",
    default="index.db",
    type=click.Path(path_type=Path),
    show_default=True,
    help="数据库目录",
)
@click.option("-n", "--limit", default=10, show_default=True, help="返回结果数量")
@click.argument("image", type=click.Path(exists=True))
def search(image: str, db_dir: Path, limit: int):
    db = IndexkusuDB(db_dir, view=True)
    _, _, _, desc = extract_descriptor(Path(image))
    for score, image in db.search_image(desc, limit):
        print(score, image)


@cli.command()
@click.option(
    "-r",
    "--regexp",
    default=".*(jpg|png|jpeg)$",
    show_default=True,
    help="匹配文件名的正则表达式（大小写不敏感）",
)
@click.option(
    "-d",
    "--db-dir",
    default="index.db",
    type=click.Path(path_type=Path),
    show_default=True,
    help="数据库目录",
)
@click.option("-t", "--threads", default=1, show_default=True, help="并发线程数")
@click.argument("PATH", type=click.Path(exists=True, path_type=Path))
def add(db_dir: Path, path: Path, regexp: str, threads: int):
    """
    计算并存储一张图片或者一个文件夹中的所有图片的特征点
    """
    re_pattern = re.compile(regexp, re.IGNORECASE)
    db = IndexkusuDB(db_dir)

    image_iter = iter([path]) if path.is_file() else path.rglob("*")
    image_list = (
        image
        for image in image_iter
        if image.is_file()
        and re_pattern.match(image.name)
        and not db.has_image(str(image))
    )

    with Pool(threads) as pool:
        results = pool.imap_unordered(extract_descriptor, image_list, chunksize=1024)
        for image, _,  _, desc in tqdm(results):
            if len(desc) != 0:
                db.add_image(str(image), desc)


def extract_descriptor(image: Path):
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, []
    img = resize_image(img)
    ft = FeatureExtractor()
    kps, desc = ft.detect_and_compute(img)
    return image, img, kps, desc


@cli.command()
@click.option(
    "-d",
    "--db-dir",
    default="index.db",
    type=click.Path(path_type=Path),
    show_default=True,
    help="数据库目录",
)
@click.option("-t", "--threads", default=1, show_default=True, help="并发线程数")
def build_index(db_dir: Path, threads: int):
    """
    构建索引
    """
    db = IndexkusuDB(db_dir)
    db.build_index(threads=threads)


@cli.command()
@click.option("--show", is_flag=True, help="展示结果图片，不进行保存")
@click.option(
    "-",
    "--output",
    default="output.png",
    type=click.Path(),
    show_default=True,
    help="将结果图片保存到指定路径",
)
@click.argument("image", type=click.Path(exists=True))
def detect(image: str, show: bool, output: str):
    """
    提取一张图片中的特征点并展示
    """
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = resize_image(img)
    ft = FeatureExtractor()
    keys, _ = ft.detect_and_compute(img)
    logger.info(f"Found {len(keys)} keypoints")    
    img = cv2.drawKeypoints(img, keys, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if show:
        cv2.imshow("result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(output, img)


@cli.command()
@click.argument("image1", type=click.Path(exists=True))
@click.argument("image2", type=click.Path(exists=True))
def match(image1: str, image2: str):
    """
    尝试匹配两张图片
    """
    _, img1, kp1, des1 = extract_descriptor(image1)
    _, img2, kp2, des2 = extract_descriptor(image2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("output.png", img3)


if __name__ == "__main__":
    cli()
