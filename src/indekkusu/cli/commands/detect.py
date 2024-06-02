import click
import cv2
from loguru import logger
from indekkusu.feature import FeatureExtractor
from .base import cli
from ..utils import load_image


@cli.command()
@click.option("--show", is_flag=True, help="展示结果图片，不进行保存")
@click.option(
    "-",
    "--output",
    default="output.png",
    type=click.Path(),
    show_default=True,
    help="将结果图片保存到指定路径",
)
@click.argument("image", type=click.Path(exists=True))
def detect(image: str, show: bool, output: str):
    """
    提取一张图片中的特征点并展示
    """
    img = load_image(image)
    keys, _ = FeatureExtractor().detect_and_compute(img)
    logger.info(f"Found {len(keys)} keypoints")
    img = cv2.drawKeypoints(
        img, keys, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    if show:
        cv2.imshow("result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(output, img)
