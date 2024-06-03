from pathlib import Path

import faiss
import numpy as np
from usearch.index import Index, MetricKind, ScalarKind
from .base import Indexer


__all__ = ["IndexkusuDB"]


class USearchIndexer(Indexer):
    def __init__(self, db_dir: Path, view: bool = False):
        if not db_dir.exists():
            db_dir.mkdir(parents=True)
        self.index = Index(
            ndim=256,
            metric=MetricKind.Hamming,
            dtype=ScalarKind.B1,
            path=db_dir / "db.usearch",
            view=view,
            enable_key_lookups=False,
        )

    def add_vectors(self, keys: np.ndarray, vectors: np.ndarray):
        self.index.add(keys, vectors)

    def search_vectors(
        self, vectors: np.ndarray, topk: int
    ) -> tuple[np.ndarray, np.ndarray]:
        result = self.index.search(vectors, topk)
        return result.keys, result.distances


class FaisssIndexer(Indexer):
    def __init__(self, db_dir: Path, view: bool = False, max_size: int = 0):
        if max_size >
        self.index = faiss.index_binary_factory(256, "BIVF1048576_HNSW32")
