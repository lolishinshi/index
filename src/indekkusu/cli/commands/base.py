from pathlib import Path

import click

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
    pass
