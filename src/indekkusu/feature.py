import numpy as np
import cv2
from cv2.typing import MatLike
from python_orb_slam3 import ORBExtractor

__all__ = ["FeatureExtractor"]


class FeatureExtractor:
    def __init__(
        self,
        n_features: int = 500,
        scale_factor: float = 1.15,
        n_levels: int = 8,
    ):
        self.orb = ORBExtractor(
            n_features=n_features,
            scale_factor=scale_factor,
            n_levels=n_levels,
            interpolation=cv2.INTER_AREA
        )

    def detect_and_compute(self, img: MatLike) -> tuple[list[cv2.KeyPoint], np.array]:
        kps, des = self.orb.detectAndCompute(img)
        if len(kps) == 0:
            return [], np.array([])
        return kps, des
