from datetime import datetime
from pathlib import Path
from collections import defaultdict
import click
from loguru import logger
from rocksdict import Rdict

from index.feature import FeatureExtractor
from index.index import FaissIndexManager
from index.utils import load_image

from .base import cli, click_db_dir


@cli.command()
@click_db_dir
@click.option("--mmap", is_flag=True, help="使用 mmap 加载索引")
@click.argument("image", type=click.Path(exists=True))
def old_search(db_dir: Path, image: str, limit: int, mmap: bool):
    """
    搜索图片
    """
    m = FaissIndexManager(db_dir)
    index = m.get_old_index(mmap)

    img = load_image(image)
    _, desc = FeatureExtractor().detect_and_compute(img)

    timestamp = datetime.now()
    result = index.search(desc, limit)
    logger.info("search time: {}", (datetime.now() - timestamp).total_seconds() * 1000)
    for k, v in result.__dict__.items():
        if k != "result":
            logger.info("{}: {}", k, v)

class ImageDB:
    def __init__(self, db_dir: Path):
        self.db = Rdict(str(db_dir / "database"))

    def find_image_path(self, feature_id: int) -> str:
        """根据特征ID查找图像路径"""
        # 先查找对应的图像ID
        image_id_bytes = self.db.get(f"id_to_image_id:{feature_id.to_bytes(8, 'little')}")
        if image_id_bytes is None:
            return None
        
        # 将图像ID转换为整数
        image_id = int.from_bytes(image_id_bytes, byteorder='little', signed=True)
        
        # 根据图像ID查找图像路径
        path_bytes = self.db.get(f"id_to_image:{image_id.to_bytes(4, 'little')}")
        if path_bytes is None:
            return None
        
        return path_bytes.decode('utf-8')