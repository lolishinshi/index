import io
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import plyvel
import numpy as np
from tqdm import tqdm
from typing import Generator
from loguru import logger
from usearch.index import Index, MetricKind, ScalarKind, BatchMatches


__all__ = ["IndexkusuDB"]

# TODO: 是否应该记录图片的哈希，这样可以避免重复添加图片
_image_prefix = b"/image"
_key_prefix = b"/key"
_vector_prefix = b"/vector"
_idx_prefix = b"/idx"


class VectorDB:
    def __init__(self, db_dir: Path):
        self.db = plyvel.DB(
            str(db_dir / "leveldb"), create_if_missing=True, compression=None
        )

    def get_key(self, image: bytes) -> int | None:
        """
        在数据库中查找图片的编号，如果不存在则返回 None
        """
        key = b"%b/%b" % (_image_prefix, image)
        if value := self.db.get(key):
            return int.from_bytes(value, "big")
        return None

    def get_image(self, key: int) -> bytes | None:
        """
        在数据库中查找图片地址，如果不存在则返回 None
        """
        bkey = b"%b/%b" % (_key_prefix, key.to_bytes(4, "big"))
        if value := self.db.get(bkey):
            return value.decode()
        return None

    def add_image(self, image: bytes, descriptors: np.ndarray):
        """
        添加一张图片及其描述子到数据库中
        """
        key = self.get_key(image)
        with self.db.write_batch() as wb:
            if key:
                key_bytes = key.to_bytes(4, "big")
            else:
                key_bytes = (
                    self.db.get(b"%b/image" % _idx_prefix) or b"\x00\x00\x00\x00"
                )
                key = int.from_bytes(key_bytes, "big")
                wb.put(b"%b/image" % _idx_prefix, (key + 1).to_bytes(4, "big"))
            wb.put(b"%b/%b" % (_image_prefix, image), key_bytes)
            wb.put(b"%b/%b" % (_key_prefix, key_bytes), image)
            wb.put(b"%b/%b" % (_vector_prefix, key_bytes), numpy_dumpb(descriptors))

    def vectors(self, start: int = 0) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        遍历数据库中的所有描述子
        """
        start_bytes = start.to_bytes(4, "big")
        with self.db.iterator(
            start=b"%b/%b" % (_vector_prefix, start_bytes), include_start=True
        ) as it:
            for key, value in it:
                yield int.from_bytes(key[8:], "big"), numpy_loadb(value)


class IndexkusuDB:
    def __init__(self, db_dir: Path, view: bool = False):
        if not db_dir.exists():
            db_dir.mkdir(parents=True)

        # TODO: connectivity expansion_add expansion_search 参数怎么设置
        self.index = Index(
            ndim=256,
            metric=MetricKind.Hamming,
            dtype=ScalarKind.B1,
            path=db_dir / "db.usearch",
            view=view,
            enable_key_lookups=False,
        )
        print(self.index.__repr_pretty__())
        self.vdb = VectorDB(db_dir)

    def has_image(self, image: str) -> bool:
        """
        判断数据库中是否已经存在该图片
        """
        return self.vdb.get_key(image.encode()) is not None

    def add_image(self, image: str, descriptors: np.ndarray):
        """
        添加一张图片及其描述子到数据库中
        """
        if descriptors is None:
            return
        self.vdb.add_image(image.encode(), descriptors)

    def build_index(self, threads: int = 1):
        """
        构建索引
        """
        for key, vector in tqdm(self.vdb.vectors()):
            # usearch 向量 id 的类型是 u32
            # 这里取前 22 bit 存放图片 id，后 10 bit 存放描述子 id
            # 理论上可以容纳 420 万张图片，每张图片 1000 个特征点
            keys = key << 10 | np.arange(0, vector.shape[0], dtype=np.uint32)
            self.index.add(keys, vector, threads=threads, copy=False)
        self.index.save()

    def search_image(
        self, descriptors: np.ndarray, topk: int = 10
    ) -> list[tuple[float, str]]:
        """
        根据描述子搜索相似图片
        """
        now = datetime.now()
        matches: BatchMatches = self.index.search(descriptors, topk)
        logger.info(f"search time: {(datetime.now() - now).microseconds / 1000}ms")
        now = datetime.now()

        match_count = defaultdict(list)
        for keys, distances in zip(matches.keys, matches.distances):
            for key, distance in zip(keys, distances):
                key = int(key)
                image = self.vdb.get_image(key >> 10)
                match_count[image].append((256 - distance) / 256)

        scores = [(wilson_score(np.array(v)), k) for k, v in match_count.items()]
        scores.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"sort time: {(datetime.now() - now).microseconds / 1000}ms")

        return scores[:topk]


# https://www.jianshu.com/p/4d2b45918958
def wilson_score(scores: np.ndarray) -> float:
    mean = np.mean(scores)
    var = np.var(scores)
    total = len(scores)
    p_z = 2.32
    score = (
        mean
        + (np.square(p_z) / (2.0 * total))
        - ((p_z / (2.0 * total)) * np.sqrt(4.0 * total * var + np.square(p_z)))
    ) / (1 + np.square(p_z) / total)
    return round(score * 100, 2)


def numpy_loadb(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


def numpy_dumpb(a: np.ndarray) -> bytes:
    with io.BytesIO() as f:
        np.save(f, a, allow_pickle=False)
        return f.getvalue()
