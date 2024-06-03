from pathlib import Path

from .metadata import connect as connect_metadata, Image
from .vector import connect as connect_vector, Vector

__all__ = ["Image", "Vector", "connect"]


def connect(path: str):
    ppath = Path(path)
    if not ppath.exists():
        ppath.mkdir(parents=True)
    connect_metadata(str(ppath / "metadata.db"))
    connect_vector(str(ppath / "vector.db"))
