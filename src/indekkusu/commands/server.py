from datetime import datetime
from pathlib import Path
from typing import Annotated

import click
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from loguru import logger

from indekkusu.database import connect, crud
from indekkusu.feature import FeatureExtractor
from indekkusu.index import FaissIndexManager
from indekkusu.utils import load_image

from .base import cli, click_db_dir

app = FastAPI()
ft = FeatureExtractor()
index = None


@app.post("/search")
async def search(
    file: Annotated[bytes, File()],
    limit: int = 10,
    k: int = 3,
    nprobe: int = 1,
    max_codes: int = 0,
):
    img = load_image(file)
    _, des = ft.detect_and_compute(img)
    result = index.search(des, limit, k, nprobe, max_codes)
    now = datetime.now()
    images = [(score, crud.image.get_by_id(id_).path) for id_, score in result.result]
    return {
        "meta": {
            "nq": result.nq,
            "nlist": result.nlist,
            "ndis": result.ndis,
            "nheap_updates": result.nheap_updates,
            "quantization_time": round(result.quantization_time, 2),
            "search_time": round(result.search_time, 2),
            "filter_time": round(result.filter_time, 2),
            "db_time": round((datetime.now() - now).total_seconds() * 100, 2),
        },
        "result": images,
    }


@cli.command()
@click_db_dir
@click.option("-n", "--name", required=True, help="索引文件名称")
@click.option("--mmap", is_flag=True, help="使用 mmap 加载索引")
@click.option("--host", default="127.0.0.1", show_default=True, help="绑定的主机地址")
@click.option("--port", default=8080, show_default=True, help="绑定的端口")
def server(db_dir: Path, name: str, mmap: bool, host: str, port: int):
    connect(str(db_dir))

    global index
    m = FaissIndexManager(db_dir)
    index = m.get_index(name, mmap)

    uvicorn.run(app, host=host, port=port)
