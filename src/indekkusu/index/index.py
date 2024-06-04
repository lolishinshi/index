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

    def add(self, key: int, vectors: np.ndarray):
        """
        添加图片的特征点向量
        """


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
