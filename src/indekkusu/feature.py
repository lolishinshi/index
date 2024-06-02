import numpy as np
from cv2 import KeyPoint, ORB
from cv2.typing import MatLike
from indekkusu.anms import ssc

__all__ = ["FeatureExtractor"]


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
        self._ft = ORB.create(
            nfeatures=nfeatures * 10, scaleFactor=scale_factor, nlevels=nlevels
        )
        self._tolerance = tolerance
        self._nfeatures = nfeatures
        self._nlevels = nlevels

        inv_scale_per_level = 1 / 1.2 ** np.arange(nlevels)
        self._feature_per_level = np.round(
            nfeatures * inv_scale_per_level**2 / np.sum(inv_scale_per_level**2)
        ).astype(np.int32)

    def detect_and_compute(self, img: MatLike) -> tuple[list[KeyPoint], np.array]:
        """
        提取图片的特征点和描述子
        """
        kps = self._ft.detect(img)
        if len(kps) == 0:
            return [], np.array([])
        if len(kps) >= self._nfeatures:
            if True:
                kp_per_level = [[] for _ in range(self._nlevels)]
                for kp in kps:
                    kp_per_level[kp.octave].append(kp)
                for i, kps in enumerate(kp_per_level):
                    if len(kps) >= self._feature_per_level[i]:
                        kp_per_level[i] = ssc(kps, self._feature_per_level[i], self._tolerance, img.shape[1], img.shape[0])
                kps = [kp for kps in kp_per_level for kp in kps]
            else:
                kps = ssc(kps, self._nfeatures, self._tolerance, img.shape[1], img.shape[0])
        kps, desc = self._ft.compute(img, kps)
        return kps, desc
