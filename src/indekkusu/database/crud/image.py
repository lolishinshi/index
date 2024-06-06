from ..metadata import Image, IndexStatus


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


def get_indexed() -> int:
    """
    返回最后一个索引的 ID
    """
    if v := IndexStatus.select().get_or_none():
        return v.indexed
    return -1


def add_indexed(add: int):
    """
    设置最后一个索引的 ID
    """
    if v := IndexStatus.select().get_or_none():
        v.indexed += add
        v.save()
    else:
        IndexStatus.create(indexed=add)
