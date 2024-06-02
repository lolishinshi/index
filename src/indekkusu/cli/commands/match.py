import click
import cv2
from indekkusu.feature import FeatureExtractor
from ..utils import load_image
from .base import cli


@cli.command()
@click.argument("image1", type=click.Path(exists=True))
@click.argument("image2", type=click.Path(exists=True))
def match(image1: str, image2: str):
    """
    尝试匹配两张图片
    """
    img1 = load_image(img1)
    img2 = load_image(img2)
    ft = FeatureExtractor()
    kp1, des1 = ft.detect_and_compute(image1)
    kp2, des2 = ft.detect_and_compute(image2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:10],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite("output.png", img3)
