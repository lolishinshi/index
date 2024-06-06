import sys
from pathlib import Path

import click
from loguru import logger

click_db_dir = click.option(
    "-d",
    "--db-dir",
    default="index.db",
    type=click.Path(path_type=Path),
    show_default=True,
    help="数据库目录",
)


@click.group()
def cli():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    )
