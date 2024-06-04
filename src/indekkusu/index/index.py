from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from .base import Indexer


class FaisssIndexer(Indexer):
    def __init__(self, db_dir: Path, view: bool = False, max_size: int = 0):
        self.db_dir = db_dir
        self.index: faiss.IndexBinaryIVF = faiss.index_binary_factory(
            256, "BIVF1048576_HNSW32"
        )

    def train(self, max_size: int, vectors: np.ndarray):
        """
        根据最大容量训练索引

        参考 https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        """
        if max_size <= 1e6:
            k = 8 * np.sqrt(max_size).astype(int)
            description = f"BIVF{k}"
        else:
            k = 2 ** (np.log10(max_size).astype(int) * 2 + 2)
            description = f"BIVF{k}_HNSW32"
        logger.info("创建索引，类型为 {}", description)
        self.index = faiss.index_binary_factory(256, description)
        self.index.verbose = True
        self.index.train(vectors)
        logger.info("训练完成")
        self.save(f"{description}.train")

    def save(self, filename: str):
        path = str(self.db_dir / filename)
        faiss.write_index_binary(self.index, path)
