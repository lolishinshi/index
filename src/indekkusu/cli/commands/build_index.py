from pathlib import Path

import click
from indekkusu.database import IndexkusuDB
from .base import cli, click_db_dir


@cli.command()
@click_db_dir
@click.option("-t", "--threads", default=1, show_default=True, help="并发线程数")
def build_index(db_dir: Path, threads: int):
    """
    构建索引
    """
    db = IndexkusuDB(db_dir)
    db.build_index(threads=threads)
