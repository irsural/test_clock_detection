from pathlib import Path

import cv2
import numpy as np

from test_clock_detection.algorithm_debugger import Debugger, DummyDebugger
from test_clock_detection.data_types import Template, Line, ClockTime
from cv2.typing import MatLike, Point
from test_clock_detection.utils import polar_to_cartesian
from test_clock_detection.data_types import MatchResultLine


def _find_line(
    src_image: MatLike,
    image_center: Point,
    angle_step_deg: float,
    min_len_line_pix: int,
    max_len_line_pix: int,
    color: tuple[int, int, int, int],
) -> list[MatchResultLine]:
    """
    Проверяет пиксели на белый цвет вокруг заданного центра. В месте с наибольшим количеством
    заданного цвета находится секундная стрелка

    :param src_image: черно-белое изображение циферблата
    :param image_center: координаты центра циферблата
    :param angle_step_deg: шаг поиска стрелки
    :param max_len_line_pix: максимальная длина искомой линии
    :param min_len_line_pix: минимальная длина искомой линии
    :param color: цвет в формате RGBA
    :return: параметры найденной секундной стрелки
    """
    match_result = []

    max_steps_angle = int(360 // angle_step_deg)
    for theta in np.linspace(start=0, stop=360, num=max_steps_angle):
        color_pixels = 0
        for radius in range(min_len_line_pix, max_len_line_pix):
            x, y = polar_to_cartesian(theta, radius, image_center, 90)

            try:
                if all(src_image[y][x] == color):
                    color_pixels += 1
            except IndexError:
                continue

        match_result.append(
            MatchResultLine(
                match_value=color_pixels, angle_deg=theta, arrow_start=image_center
            )
        )

    return match_result

def _find_best_lines(
    src_image: MatLike,
    image_center: Point,
    angle_step_deg: float,
    min_len_line_pix: int,
    max_len_line_pix: int,
    color: tuple[int, int, int, int],
    ) -> list[Line]:
    """
    Поиск 3-х лучших цветных линий исходящий из заданного центра изображения

    :param src_image: изображение в формате BGRA
    :param image_center: центр изображения
    :param angle_step_deg: шаг поиска линии
    :param min_len_line_pix: минимальная длина линии
    :param max_len_line_pix: максимальная длина линии
    :param color: цвет линии
    :return: список с 3 лучшими совпадениями линий
    """
    match_result = _find_line(src_image, image_center, angle_step_deg, min_len_line_pix, max_len_line_pix, color)
    match_result.sort(reverse=True, key=lambda x: x.match_value)

    lines = []
    for index, match in enumerate(match_result[:3]):
        lines.append(Line(name=f'{index + 1} линяя',
                          line_start=match.arrow_start,
                          angle_deg=match.angle_deg,
                          len_line=max_len_line_pix))
    return lines


def detect_time(
    root_folder: Path, image_path: Path, debug_mode: None | Debugger = None
) -> ClockTime:
    """
    Скрипт определения времени на часах.
    Ниже приведены примеры как пользоваться дебагером, чтобы сохранять промежуточные результаты
    работы алгоритма.

    :param image_path: корневая папка проекта. Может быть полезна для загрузки дополнительных
      артефактов работы алгоритма, например шаблонов стрелок и т. д.
    :param image_path: путь к изображению часов
    :param debug_mode: режим отладки
    :return: время на часах в формате чч:мм:сс.мс
    """
    debugger = debug_mode if debug_mode is not None else DummyDebugger()

    image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debugger.save_image('Серое изображение', image_gray)

    # Поиск линий на изображении
    image_center = (315, 250)
    image_gray_rgba = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGRA)
    best_lines = _find_best_lines(image_gray_rgba, image_center, 1, 0, 200, (255, 255, 255, 255))
    # Отрисовка линий на оригинальном изображении
    debugger.save_image_with_contours('Линия из центра изображения', image_gray_rgba, None, best_lines)

    # Сохранение результата алгоритма определения времени
    result_time = ClockTime(hours=0, minutes=0, seconds=0, ms=0)
    return result_time
