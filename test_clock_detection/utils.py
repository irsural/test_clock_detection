from datetime import datetime

import numpy as np
from cv2.typing import Point


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


def check_result(
    excepted_time_ms: datetime, result_time_ms: datetime, accuracy_sec: float
) -> tuple[float, bool]:
    """
    Проверяет успешность проведенного теста. Если разница между определенным и фактическим
    временем на изображении составила больше **accuracy** сек, то такой результат
    считается неудачным

    :param excepted_time_ms: фактическое время на изображении
    :param result_time_ms: определенное время алгоритмом
    :param accuracy_sec: максимальное отклонение от реального значения, после которого
      определение времени считается неудачным. Задается в секундах
    :return: разницу между фактическим и определенным временем и является ли тест успешным
    """
    delta = abs(excepted_time_ms - result_time_ms)
    delta_sec = round(float(delta.total_seconds()), 1)

    if delta_sec <= accuracy_sec:
        return delta_sec, True
    else:
        return delta_sec, False
