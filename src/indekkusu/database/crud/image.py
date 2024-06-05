from ..metadata import Image, IndexStatus, VectorNumber


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


def get_indexed() -> int:
    """
    返回最后一个索引的 ID
    """
    return IndexStatus.select().get().indexed


def set_indexed(indexed: int):
    """
    设置最后一个索引的 ID
    """
    IndexStatus.update(indexed=indexed).execute()
