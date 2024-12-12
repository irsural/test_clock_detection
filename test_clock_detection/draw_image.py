import cv2
from cv2.typing import MatLike, Point

from test_clock_detection.data_types import Template


def draw_line(image: MatLike, start_line: Point, end_line: Point) -> MatLike:
    """
    Рисует на изображении цветную красную линию

    :param image: изображение в формате RGBA
    :param start_line: координаты начала линии
    :param end_line: координаты конца линии
    :return:
    """
    cv2.line(image, start_line, end_line, (0, 0, 255, 255), 1)
    return image
