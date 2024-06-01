import cv2
import numpy as np
from cv2 import KeyPoint, ORB, FastFeatureDetector
from cv2.typing import MatLike
from indekkusu.anms import ssc


class FeatureExtractor:
    def __init__(
        self,
        nfeatures: int = 500,
        tolerance: float = 0.05,
        scale_factor: float = 1.2,
        nlevels: int = 8,
    ):
        """
        初始化特征提取器

        :param nfeatures: 最大特征点数量
        :param tolerance: ANMS 算法的容差
        :param scale_factor: 图像金字塔的缩放因子
        :param nlevels: 图像金字塔的层数
        """
        self._ft = ORB.create(nfeatures=nfeatures * 10, scaleFactor=scale_factor, nlevels=nlevels)
        self._tolerance = tolerance
        self._nfeatures = nfeatures

    def detect_and_compute(self, img: MatLike) -> tuple[list[KeyPoint], np.array]:
        """
        提取图片的特征点和描述子
        """
        kps = self._ft.detect(img)
        if len(kps) == 0:
            return [], np.array([])
        kps = ssc(kps, self._nfeatures, self._tolerance, img.shape[1], img.shape[0])
        kps, desc = self._ft.compute(img, kps)
        return kps, desc

