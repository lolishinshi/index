import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from loguru import logger


@dataclass
class FaissSearchResult:
    nq: int
    nlist: int
    ndis: int
    nheap_updates: int
    quantization_time: float
    search_time: float
    result: list[tuple[int, float]]


# TODO: 阻止并发访问
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
        limit: int = 10,
        k: int = 3,
        nprobe: int = 1,
        max_codes: int = 0,
        efSearch: int = 16,
    ) -> FaissSearchResult:
        """
        搜索最近的特征点，返回图片 ID 和距离
        """
        self.index.nprobe = nprobe
        self.index.max_codes = max_codes
        distances, labels = self.index.search(vectors, k)

        labels >>= 10
        kds = defaultdict(list)
        for label, distance in zip(labels, distances):
            # 如果某个特征点匹配到了同一图片中的多个特征点，只取最接近的一个
            t = defaultdict(lambda: 256)
            for l, d in zip(label, distance):
                t[l] = min(t[l], d)
            for l, d in t.items():
                kds[l].append(d)

        # 此处先按长度截取前 2 * limit 个，减少计算量
        kls = sorted(kds.items(), key=lambda x: len(x[1]), reverse=True)
        kws = sorted(
            [(k, wilson_score(np.array(v))) for k, v in kls[: 2 * limit]],
            key=lambda x: x[1],
            reverse=True,
        )

        result = FaissSearchResult(
            nq=faiss.cvar.indexIVF_stats.nq,
            nlist=faiss.cvar.indexIVF_stats.nlist,
            ndis=faiss.cvar.indexIVF_stats.ndis,
            nheap_updates=faiss.cvar.indexIVF_stats.nheap_updates,
            quantization_time=faiss.cvar.indexIVF_stats.quantization_time,
            search_time=faiss.cvar.indexIVF_stats.search_time,
            result=kws[:limit],
        )
        faiss.cvar.indexIVF_stats.reset()
        return result

    def save(self):
        """
        保存索引
        """
        faiss.write_index_binary(self.index, self.path + ".tmp")
        shutil.move(self.path + ".tmp", self.path)


class FaissIndexManager:
    def __init__(self, db_dir: Path, description: str | None = None):
        self.db_dir = db_dir
        self.description = description

    def get_index(self, index_name: str, mmap: bool = False) -> FaissIndex:
        trained_path = self.db_dir / f"{self.description}.train"

        if any(self.db_dir.glob(f"*.index.{index_name}")):
            index_path = next(self.db_dir.glob(f"*.index.{index_name}"))
        elif (self.db_dir / f"{self.description}.train").exists():
            index_path = self.db_dir / f"{self.description}.index.{index_name}"
            shutil.copy(trained_path, index_path)
        else:
            raise FileNotFoundError(f"索引文件 {index_name} 不存在")

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
