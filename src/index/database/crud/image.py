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


def get_indexed(name: str) -> int:
    """
    返回已索引的图片数量
    """
    if v := IndexStatus.select().where(IndexStatus.name == name).get_or_none():
        return v.indexed
    return IndexStatus.create(name=name, indexed=0).indexed


def add_indexed(name: str, add: int):
    """
    将已索引的图片数量增加 add
    """
    IndexStatus.update(indexed=IndexStatus.indexed + add).where(
        IndexStatus.name == name
    ).execute()
