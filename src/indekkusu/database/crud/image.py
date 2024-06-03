from blake3 import blake3
from ..metadata import Image


def create(path: str) -> int | None:
    """
    记录图片路径，返回其 id
    如果图片已存在则返回 None
    """
    with open(path, "rb") as f:
        hash = blake3(f.read()).digest()
    image = Image.get_or_none(Image.hash == hash)
    if image is None:
        image = Image.create(hash=hash, path=path)
        return image.id
    else:
        return None


def get_by_id(image_id: int) -> Image:
    """
    通过ID查询图片
    """
    return Image.get_by_id(image_id)
