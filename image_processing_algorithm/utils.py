from dataclasses import dataclass
import numpy as np
from cv2.typing import MatLike, Point


@dataclass
class Template:
    """Данные о шаблонах"""

    name: str
    """ Имя Шаблона """
    image: MatLike
    """ Изображение шаблона """
    top_left: Point
    """ Координата левого-верхнего угла """
    angle_deg: float
    """ Угол поворота шаблона в градусах """


@dataclass
class Line:
    """Данные о линии"""

    name: str
    """ Имя линии """
    line_start: Point
    """ Координаты начала линии """
    angle_deg: float
    """ Угол поворота линии в градусах """
    len_line: int
    """ Длина линии """


def polar_to_cartesian(
    angle_deg: float, radius: int, center: Point, angle_offset_deg: float = 0
) -> Point:
    """
    Переводит полярные координаты в координаты пикселей на изображении

    :param angle_deg: угол в градусах
    :param radius: радиус
    :param center: координаты центра
    :param angle_offset_deg: смещение начального угла
    :return:
    """
    x = round(center[0] - radius * np.cos(np.radians(angle_deg + angle_offset_deg)))
    y = round(center[1] - radius * np.sin(np.radians(angle_deg + angle_offset_deg)))

    return x, y
