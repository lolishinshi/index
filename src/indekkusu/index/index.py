from pathlib import Path

import faiss
import numpy as np
from loguru import logger


class FaissIndex:
    def __init__(self, path: str, mmap: bool = False):
        """
        加载索引
        """
        if mmap:
            io_flags = faiss.IO_FLAG_READ_ONLY | faiss.IO_FLAG_MMAP
        else:
            io_flags = 0
        self.path = path
        self.index: faiss.IndexBinaryIVF = faiss.read_index_binary(path, io_flags)

    def imbalance(self) -> float:
        """
        当前索引的不平衡度，1 为绝对平均
        """
        arr = np.array([self.index.get_list_size(i) for i in range(self.index.nlist)])
        uf = np.sum(arr.astype(np.float64) ** 2)
        tot = np.sum(arr).astype(np.float64)
        return float(uf * len(arr) / tot**2)

    def add(self, vectors: np.ndarray):
        """
        添加图片的特征点向量
        """
        self.index.add(vectors)

    def search(
        self,
        vectors: np.ndarray,
        k: int,
        nprobe: int = 1,
        max_codes: int = 0,
        efSearch: int = 16,
    ) -> np.ndarray:
        """
        搜索最近的向量
        """
        params = faiss.SearchParametersIVF(
            nprobe=nprobe,
            max_codes=max_codes,
            quantizer_params=faiss.SearchParametersHNSW(efSearch=efSearch),
        )
        return self.index.search(vectors, k, params=params)

    def save(self):
        """
        保存索引
        """
        faiss.write_index_binary(self.index, self.path)


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
