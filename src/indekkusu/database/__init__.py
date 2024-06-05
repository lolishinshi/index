from pathlib import Path

from .metadata import Image
from .metadata import connect as connect_metadata
from .vector import Vector
from .vector import connect as connect_vector

__all__ = ["Image", "Vector", "connect"]


def connect(
    path: str, metadata: bool = True, vector: bool = True, readonly: bool = False
):
    ppath = Path(path)
    if not ppath.exists():
        ppath.mkdir(parents=True)
    if metadata:
        connect_metadata(str(ppath / "metadata.db"), readonly)
    if vector:
        connect_vector(str(ppath / "vector.db"), readonly)
