from pathlib import Path

import click

from indekkusu.index.FaissIndexTrainer import FaissIndexTrainer

from .base import cli, click_db_dir


@cli.command()
@click_db_dir
def build_index(db_dir: Path, image_num: int, feature_num: int):
    """
    构建索引
    """
    pass
