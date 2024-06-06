import shutil
from pathlib import Path
from collections import defaultdict

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

    def add_with_ids(self, vectors: np.ndarray, xids: np.ndarray):
        """
        添加图片的特征点向量
        """
        self.index.add_with_ids(vectors, xids)

    def search(
        self,
        vectors: np.ndarray,
        k: int,
        nprobe: int = 1,
        max_codes: int = 0,
        efSearch: int = 16,
    ) -> dict[int, float]:
        """
        搜索最近的特征点，返回图片 ID 和距离
        """
        params = faiss.SearchParametersIVF(
            nprobe=nprobe,
            max_codes=max_codes,
            quantizer_params=faiss.SearchParametersHNSW(efSearch=efSearch),
        )
        # TODO: 为什么不能设置 params
        distances, labels = self.index.search(vectors, k)

        labels >>= 10
        result = defaultdict(list)
        for label, distance in zip(labels, distances):
            t = defaultdict(lambda: 256)
            # 如果某个特征点匹配到了同一图片中的多个特征点，只取最接近的一个
            for l, d in zip(label, distance):
                t[l] = min(t[l], d)
            for l, d in t.items():
                result[l].append(d)

        return {k: wilson_score(np.array(v)) for k, v in result.items()}

    def save(self):
        """
        保存索引
        """
        faiss.write_index_binary(self.index, self.path + ".tmp")
        shutil.move(self.path + ".tmp", self.path)


class FaissIndexManager:
    def __init__(self, db_dir: Path, description: str):
        self.db_dir = db_dir
        self.description = description

    def get_index(self, index_name: str, mmap: bool = False) -> FaissIndex:
        index_path = self.db_dir / f"{self.description}.index.{index_name}"
        trained_path = self.db_dir / f"{self.description}.train"
        if not index_path.exists():
            shutil.copy(trained_path, index_path)
        return FaissIndex(str(index_path), mmap)


# https://www.jianshu.com/p/4d2b45918958
def wilson_score(scores: np.ndarray) -> float:
    scores = 1 - scores / 256
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
