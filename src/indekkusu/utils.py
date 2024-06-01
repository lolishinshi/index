import cv2
from cv2.typing import MatLike


def resize_image(img: MatLike, width: int = 960) -> MatLike:
    """
    将图片按照宽度等比例缩放
    """
    _, w = img.shape[:2]
    scale = width / w
    if w <= width:
        return img
    else:
        return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
