from abc import ABC, abstractmethod

import numpy as np


class SearchResult:
    time: float
    keys: np.ndarray
    scores: np.ndarray


class Indexer(ABC):
    @abstractmethod
    def add_vectors(self, keys: np.ndarray, vectors: np.ndarray):
        """
        批量添加向量

        :param keys: 向量的 key，长度为 n
        :param vectors: 向量，形状为 (n, 256)
        """
        pass

    @abstractmethod
    def search_vectors(
        self, vectors: np.ndarray, topk: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        批量搜索向量

        :param vectors: 向量，形状为 (n, 256)
        :param topk: 返回的最大数量
        :return: keys, distances
        """
        pass


# https://www.jianshu.com/p/4d2b45918958
def wilson_score(scores: np.ndarray) -> float:
    mean = np.mean(scores)
    var = np.var(scores)
    total = len(scores)
    p_z = 1.98
    score = (
        mean
        + (np.square(p_z) / (2.0 * total))
        - ((p_z / (2.0 * total)) * np.sqrt(4.0 * total * var + np.square(p_z)))
    ) / (1 + np.square(p_z) / total)
    return round(score * 100, 2)
