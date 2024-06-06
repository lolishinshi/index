from pathlib import Path

import click

from indekkusu.feature import FeatureExtractor
from indekkusu.database import connect
from indekkusu.index import FaissIndexManager

from ..utils import load_image
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
    print(result)
    # 注：如果某张图片中含有大量重复图形，可能会导致提取出多个相似的特征点
    # 如果恰好有图片也有这个特征，那么这个垃圾图片的得分会很高，应该限制一个特征点的匹配结果中，每张图片只能出现一次
