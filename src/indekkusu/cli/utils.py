from pathlib import Path

import numpy as np
import cv2
from cv2.typing import MatLike


def load_image(image: Path | str | bytes) -> MatLike | None:
    """
    读取图片，转换为灰度图并缩放
    """
    if isinstance(image, bytes):
        buf = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = resize_image(img)
    return img


def resize_image(img: MatLike, width: int = 1080, height: int = 1920) -> MatLike:
    """
    将图片按照宽度等比例缩放
    """
    h, w = img.shape[:2]
    if w <= width and h <= height:
        return img
    else:
        scale = min(width / w, height / h)
        return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
