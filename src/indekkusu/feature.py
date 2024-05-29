import cv2
from cv2 import KeyPoint, Feature2D, ORB, FastFeatureDetector
from cv2.typing import MatLike
from indekkusu.anms import ssc


class FeatureExtractor:
    ds: Feature2D
    ft: Feature2D
    tolerance = 0.1

    def __init__(self):
        self.ds = FastFeatureDetector.create()
        self.ft = ORB.create()

    def detect(self, img: MatLike, limit: int = 500) -> list[KeyPoint]:
        """
        提取图片的特征点，返回特征点列表
        """
        keys = self.ds.detect(img)
        return ssc(keys, limit, self.tolerance, img.shape[1], img.shape[0])

    def detect_and_compute(
        self, img: MatLike, limit: int = 500, resize: int | None = 1280
    ) -> tuple[MatLike, list[KeyPoint], MatLike]:
        """
        提取图片的特征点和描述子，返回缩放后的图片、特征点、描述子
        """
        if resize:
            img = resize_image(img, resize)
        keys = self.detect(img, limit)
        _, desc = self.ft.compute(img, keys)
        return img, keys, desc


def resize_image(img: MatLike, width: int) -> MatLike:
    """
    将图片按照宽度等比例缩放
    """
    _, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
