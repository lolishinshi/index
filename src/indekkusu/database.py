import os
import plyvel
import numpy as np
from collections import defaultdict
from usearch.index import Index, MetricKind, ScalarKind, BatchMatches


class IndexkusuDB:
    def __init__(self, db_dir: str):
        # TODO: connectivity expansion_add expansion_search 参数怎么设置
        self.index = Index(
            ndim=256, metric=MetricKind.Hamming, dtype=ScalarKind.B1, multi=True
        )
        if os.path.exists(f"{db_dir}/db.usearch"):
            self.index.load(f"{db_dir}/db.usearch")
        self.image_db = plyvel.DB(f"{db_dir}/image.ldb", create_if_missing=True)
        self.key_db = plyvel.DB(f"{db_dir}/key.ldb", create_if_missing=True)
        self.db_dir = db_dir

    def has_image(self, image: str) -> bool:
        """
        判断数据库中是否已经存在该图片
        """
        return self.image_db.get(image.encode()) is not None

    def _add_image(self, image: str) -> int:
        """
        添加图片路径到数据库中，返回图片的编号
        """
        with self.image_db.write_batch() as wb:
            idx = self.image_db.get(b"__idx__")
            if idx is None:
                idx = b"0"
            else:
                idx = str(int(idx) + 1).encode()
            wb.put(image.encode(), idx)
            wb.put(b"__idx__", idx)
        self.key_db.put(idx, image.encode())
        return int(idx)

    def add_image(self, image: str, descriptors: np.ndarray):
        """
        添加一张图片及其描述子到数据库中
        """
        if self.has_image(image):
            return
        assert descriptors.shape[1] == 32
        key = self._add_image(image)
        keys = np.array([key] * descriptors.shape[0])
        self.index.add(keys, descriptors, copy=False)

    def search_image(self, descriptors: np.ndarray, limit: int = 10) -> list[str]:
        matches: BatchMatches = self.index.search(descriptors, limit)
        scores = defaultdict(list)
        for keys, distances in zip(matches.keys, matches.distances):
            for key, distance in zip(keys, distances):
                scores[key].append((128 - distance) / 128)

        scores = [(k, wilson_score(np.array(v))) for k, v in scores.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        for k, v in scores[:10]:
            print(f"{self.key_db.get(str(k).encode())}: {v}")

    def close(self):
        self.index.save(f"{self.db_dir}/db.usearch")
        self.image_db.close()


# https://www.jianshu.com/p/4d2b45918958
def wilson_score(scores: np.ndarray):
    mean = np.mean(scores)
    var = np.var(scores)
    total = len(scores)
    p_z = 2.32
    score = (
        mean
        + (np.square(p_z) / (2.0 * total))
        - ((p_z / (2.0 * total)) * np.sqrt(4.0 * total * var + np.square(p_z)))
    ) / (1 + np.square(p_z) / total)
    return score
