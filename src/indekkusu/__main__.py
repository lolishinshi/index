import re
import cv2
import click
from indekkusu.feature import FeatureExtractor
from indekkusu.database import IndexkusuDB
from pathlib import Path
from tqdm import tqdm


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--db-dir",
    default="index.db",
    type=click.Path(exists=True),
    show_default=True,
    help="数据库目录",
)
@click.option("--limit", default=10, show_default=True, help="返回结果数量")
@click.argument("image", type=click.Path(exists=True))
def search(image: str, db_dir: str, limit: int = 10):
    db = IndexkusuDB(db_dir)
    img = cv2.imread(image)
    ft = FeatureExtractor()
    _, _, desc = ft.detect_and_compute(img)
    db.search_image(desc)


@cli.command()
@click.option(
    "--regexp",
    default=".*(jpg|png|jpeg)$",
    show_default=True,
    help="匹配文件名的正则表达式（大小写不敏感）",
)
@click.option(
    "--db-dir",
    default="index.db",
    type=click.Path(path_type=Path),
    show_default=True,
    help="数据库目录",
)
@click.argument("PATH", type=click.Path(exists=True, path_type=Path))
def add(db_dir: Path, path: Path, regexp: str):
    """
    索引一张图片或者一个文件夹中的所有图片
    """
    re_pattern = re.compile(regexp, re.IGNORECASE)

    if path.is_file():
        images = iter([path])
    else:
        images = path.glob("*")

    if not db_dir.exists():
        db_dir.mkdir(parents=True)

    db = IndexkusuDB(str(db_dir))

    for image in tqdm(images):
        if image.is_file() and re_pattern.match(image.name):
            img = cv2.imread(str(image))
            ft = FeatureExtractor()
            _, _, desc = ft.detect_and_compute(img)
            db.add_image(str(image), desc)

    db.close()


@cli.command()
@click.option("--show", is_flag=True, help="展示结果图片，不进行保存")
@click.option(
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
    img = cv2.imread(image)
    ft = FeatureExtractor()
    img, keys, _ = ft.detect_and_compute(img)
    img = cv2.drawKeypoints(img, keys, None)
    if show:
        cv2.imshow("result", img)
        while cv2.waitKey() != -1:
            pass
    else:
        cv2.imwrite(output, img)


if __name__ == "__main__":
    cli()
