from pathlib import Path


import click
import cv2
from tqdm import tqdm
from prettytable import PrettyTable
from indekkusu.database import IndexkusuDB
from indekkusu.feature import FeatureExtractor
from indekkusu.cli.commands.base import click_db_dir


@click.command()
@click_db_dir
@click.argument("search", type=click.Path(exists=True, path_type=Path))
def gen_report(db_dir: Path, search: Path):
    """
    生成搜索效果检测报告
    """
    db = IndexkusuDB(db_dir)
    ft = FeatureExtractor()

    table = PrettyTable(["测试集", "正确率"])

    for testset in search.iterdir():
        print(f"Testset: {testset.name}")

        stats = {"success": 0, "total": 0}
        for image in tqdm(testset.glob("*.*")):
            img = cv2.imread(str(image))
            if img is None:
                continue
            _, desc = ft.detect_and_compute(img)
            if len(desc) == 0:
                continue
            try:
                result = db.search_image(desc, 3)
                if image.stem.startswith(Path(result[0][1]).stem):
                    stats["success"] += 1
                stats["total"] += 1
            except:
                pass

        table.add_row([testset.name, round(stats["success"] / stats["total"] * 100, 2)])
        print(stats)

    print(table)


if __name__ == "__main__":
    gen_report()