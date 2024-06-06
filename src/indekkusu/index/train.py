from pathlib import Path

import faiss
import numpy as np


class FaissIndexTrainer:
    d = 256

    def __init__(self, db_dir: Path, max_size: int, description: str | None = None):
        # 参考 https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        if description is None:
            if max_size <= 1e6:
                k = 8 * np.sqrt(max_size).astype(int)
                description = f"BIVF{k}"
            else:
                k = 2 ** (np.log10(max_size).astype(int) * 2 + 2)
                description = f"BIVF{k}_HNSW32"
        self.db_dir = db_dir
        self.description = description
        self.index: faiss.IndexBinaryIVF = faiss.index_binary_factory(
            self.d, description
        )
        self.k = self.index.nlist
        self.index.verbose = True
        self.index.cp.verbose = True

    def train(self, vectors: np.ndarray):
        """
        训练索引
        """
        self.index.train(vectors)

    def train_gpu(self, vectors: np.ndarray):
        """
        使用 GPU 训练索引
        """
        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(self.d))
        self.index.clustering_index = clustering_index
        self.index.train(vectors)

    def save(self):
        path = self.db_dir / f"{self.description}.train"
        faiss.write_index_binary(self.index, str(path))
