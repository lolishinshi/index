import cv2
import numpy as np
from cv2 import KeyPoint, ORB
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
        self._ft = ORB.create(nfeatures=nfeatures * 10, nlevels=1)
        self._nfeatures = nfeatures
        self._tolerance = tolerance
        self._scale_factor = scale_factor
        self._nlevels = nlevels

        self._scale_factors = scale_factor ** np.arange(nlevels)
        self._inv_scale_factors = 1 / self._scale_factors
        self._features_per_level = np.round(
            nfeatures
            * np.square(self._inv_scale_factors)
            / np.linalg.norm(self._inv_scale_factors) ** 2
        ).astype(np.int32)

    def _compute_pyramid(self, img: MatLike) -> list[MatLike]:
        """
        计算图像金字塔
        """
        pyramid = [img]
        for i in range(1, self._nlevels):
            nimg = cv2.resize(
                img,
                (0, 0),
                fx=self._inv_scale_factors[i],
                fy=self._inv_scale_factors[i],
                interpolation=cv2.INTER_AREA,
            )
            # 参照 slam3 的算法修改 https://blog.csdn.net/weixin_45947476/article/details/123738789
            pyramid.append(nimg)
        return pyramid

    def detect_and_compute(self, img: MatLike) -> tuple[list[KeyPoint], np.array]:
        """
        提取图片的特征点和描述子
        """
        pyramid = self._compute_pyramid(img)
        keypoints = []
        descriptors = []
        for i, img in enumerate(pyramid):
            kp = self._ft.detect(img)
            kp = ssc(kp, self._features_per_level[i], self._tolerance, img.shape[1], img.shape[0])
            kp, desc = self._ft.compute(img, kp)
            for k in kp:
                k.pt = (k.pt[0] * self._scale_factors[i], k.pt[1] * self._scale_factors[i])
                k.size *= self._scale_factors[i]
            keypoints.extend(kp)
            descriptors.append(desc)
        return keypoints, np.vstack(descriptors)

