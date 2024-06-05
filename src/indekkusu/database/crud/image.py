from blake3 import blake3

from ..metadata import Image, VectorNumber


def create(hash: bytes, path: str) -> int:
    """
    记录图片路径，返回其 id
    """
    return Image.create(hash=hash, path=path).id


def check_hash(hash: bytes) -> bool:
    """
    检查图片是否已存在
    """
    return Image.get_or_none(Image.hash == hash) is not None


def get_by_id(image_id: int) -> Image:
    """
    通过ID查询图片
    """
    return Image.get_by_id(image_id)


def add_vector_num(image_id: int, vector_num: int):
    """
    记录特征点数量
    """
    VectorNumber.create(id=image_id, num=vector_num)
