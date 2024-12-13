from pathlib import Path

import cv2
from cv2.typing import MatLike, Point
import numpy as np

from test_clock_detection.algorithm_debugger import Debugger, DummyDebugger
from test_clock_detection.data_types import ClockTime, Line, MatchResultLine
from test_clock_detection.utils import polar_to_cartesian


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
            MatchResultLine(match_value=color_pixels, angle_deg=theta, arrow_start=image_center)
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
    Поиск 3-х лучших цветных линий исходящий из заданного центра изображения, разница углов между
    которыми превышает 30 градусов

    :param src_image: изображение в формате BGRA
    :param image_center: центр изображения
    :param angle_step_deg: шаг поиска линии
    :param min_len_line_pix: минимальная длина линии
    :param max_len_line_pix: максимальная длина линии
    :param color: цвет линии
    :return: список с 3 лучшими совпадениями линий
    """
    match_result = _find_line(
        src_image, image_center, angle_step_deg, min_len_line_pix, max_len_line_pix, color
    )
    match_result.sort(reverse=True, key=lambda x: x.match_value)

    lines: list[Line] = []
    count_lines = 0
    lines_angles: list[float] = []
    for index, match in enumerate(match_result):
        if index != 0:
            correct_angle_diff = True
            for line in lines:
                if abs(line.angle_deg - match.angle_deg) < 30:
                    correct_angle_diff = False
            if not correct_angle_diff:
                continue
        lines.append(
            Line(
                name=f'{index + 1} линяя',
                line_start=match.arrow_start,
                angle_deg=match.angle_deg,
                len_line=max_len_line_pix,
            )
        )
        lines_angles.append(match.angle_deg)
        count_lines += 1
        if count_lines == 3:
            break
    return lines


def _angle_to_hours(arrow_angle: float) -> int:
    """
    Перевод угла поворота стрелки в часы

    :param arrow_angle: угол поворота стрелки
    :return: значение стрелки в часах
    """
    hours = arrow_angle // 30
    return int(hours % 12)


def _angle_to_minutes_seconds(arrow_angle: float) -> int:
    """
    Перевод угла поворота стрелки в секунды или минуты

    :param arrow_angle: угол поворота стрелки
    :return: значение стрелки в минутах или секундах
    """
    min_sec = arrow_angle // 6
    return int(min_sec % 60)


def _angles_to_ms(arrow_angle: float) -> int:
    """
    Перевод значения стрелки в миллисекунды

    :param arrow_angle: угол поворота стрелки
    :return: значение стрелки в миллисекундах
    """
    milliseconds = int(arrow_angle / 0.006) % 1000
    return milliseconds


def _convert_angle_to_time(
    hours_arrow: Line, minutes_arrow: Line, seconds_arrow: Line
) -> ClockTime:
    """
    Перевод углов поворота стрелок в значения времени

    :param hours_arrow: часовая стрелка
    :param minutes_arrow: минутная стрелка
    :param seconds_arrow: секундная стрелка
    :return: время часов
    """
    hours = _angle_to_hours(hours_arrow.angle_deg)
    minutes = _angle_to_minutes_seconds(minutes_arrow.angle_deg)
    seconds = _angle_to_minutes_seconds(seconds_arrow.angle_deg)
    milliseconds = _angles_to_ms(seconds_arrow.angle_deg)

    return ClockTime(hours=hours, minutes=minutes, seconds=seconds, ms=milliseconds)


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
    image_binary = cv2.threshold(image_gray, 220, 255, cv2.THRESH_BINARY)[1]
    debugger.save_image('Бинарное изображение', image_binary)
    image_binary_rgba = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2BGRA)

    image_center = (315, 250)
    best_lines = _find_best_lines(image_binary_rgba, image_center, 1, 0, 200, (255, 255, 255, 255))
    # Отрисовка линий на оригинальном изображении
    debugger.save_image_with_lines('Линия из центра изображения', image_binary_rgba, best_lines)

    # Сохранение результата алгоритма определения времени
    result_time = _convert_angle_to_time(best_lines[2], best_lines[1], best_lines[0])
    return result_time
