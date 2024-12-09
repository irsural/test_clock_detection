from dataclasses import dataclass
from datetime import datetime
import re

import numpy as np
from cv2.typing import MatLike, Point
from typing_extensions import Self


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


@dataclass(kw_only=True)
class ClockTime:
    hours: int
    minutes: int
    seconds: int
    ms: int

    def __str__(self) -> str:
        """
        Форматирует строку в вид: чч:мм:сс.мс

        :return: отформатированная строка
        """
        return f'{self.hours:02}:{self.minutes:02}:{self.seconds:02}.{self.ms:03}'

    @classmethod
    def from_ms(cls, clock_time_ms: float) -> Self:
        """
        Разбивает время в мс на часы, минуты, секунды и миллисекунды

        :param clock_time_ms: время пройденное часами с начала видео
        были пройти к заданному моменту
        :return: значение времени приведенное к часам, минутам и секундам
        """

        milliseconds = int(clock_time_ms % 1000)
        seconds = int((clock_time_ms // 1000) % 60)
        minutes = int((clock_time_ms // 1000 // 60) % 60)
        hours = int(clock_time_ms // 1000 // 60 // 60)

        return cls(hours=hours, minutes=minutes, seconds=seconds, ms=milliseconds)


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


RESULT_IMAGE_REGULAR = re.compile(
    r'^(?P<error_status>[01]+)-(?P<error>\d+\.\d+)-(?P<detected_time>\d{2}:\d{2}:\d{2}\.\d{3})-'
    r'(?P<excepted_time>\d{2}:\d{2}:\d{2}\.\d{3})-(?P<detected_clock_angle>\d+\.\d+)-'
    r'(?P<real_clock_angle>\d+)$'
)


@dataclass
class DetectTimeResult:
    success_detection: bool
    error_sec: float
    detected_time: datetime
    excepted_time: datetime

    def to_str(self) -> str:
        """
        Формирует строку из результатов алгоритма определения времени по аналоговым часам

        :return: строку из результатов алгоритма определения времени
        """

        return (
            f'{self.success_detection}-{self.error_sec}-{self.detected_time.strftime("%H:%M:%S.%f")[:-3]}-'
            f'{self.excepted_time.strftime("%H:%M:%S.%f")[:-3]}-'
        )

    @classmethod
    def from_str(cls, result_of_algorithm: str) -> Self:
        """
        Проверяет строку на соответствие формату результатов алгоритма определения времени по
        аналоговым часам.

        :param result_of_algorithm: строка с результатами алгоритма определения времени по
        аналоговым часам
        :return: параметры результатов работы алгоритма определения времени по аналоговым часам
        """

        result_attr = RESULT_IMAGE_REGULAR.search(result_of_algorithm)
        assert (
            result_attr is not None
        ), f'Строка: {result_of_algorithm} - не соответствует формату результатов алгоритма'

        return cls(
            success_detection=int(result_attr.group('error_status')),
            error_sec=float(result_attr.group('error')),
            detected_time=datetime.strptime(result_attr.group('detected_time'), '%H:%M:%S.%f'),
            excepted_time=datetime.strptime(result_attr.group('excepted_time'), '%H:%M:%S.%f'),
        )
