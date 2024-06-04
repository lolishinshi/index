from pathlib import Path

import click

from indekkusu.feature import FeatureExtractor

from ..utils import load_image
from .base import cli, click_db_dir


@cli.command()
@click_db_dir
@click.option("-n", "--limit", default=10, show_default=True, help="返回结果数量")
@click.argument("image", type=click.Path(exists=True))
def search(image: str, db_dir: Path, limit: int):
    db = IndexkusuDB(db_dir, view=True)
    img = load_image(image)
    _, desc = FeatureExtractor().detect_and_compute(img)
    for score, image in db.search_image(desc, limit):
        print(score, image)
