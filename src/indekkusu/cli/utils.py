from pathlib import Path

import cv2
from cv2.typing import MatLike


def load_image(image: Path | str) -> MatLike | None:
    """
    读取图片，转换为灰度图并缩放
    """
    img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = resize_image(img)
    return img


def resize_image(img: MatLike, width: int = 1280) -> MatLike:
    """
    将图片按照宽度等比例缩放
    """
    _, w = img.shape[:2]
    if w <= width:
        return img
    else:
        scale = width / w
        return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
