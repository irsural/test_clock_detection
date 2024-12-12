import re
from dataclasses import dataclass
from datetime import datetime

from cv2.typing import Point
from typing_extensions import Self


@dataclass
class MatchResultLine:
    """Данные о совпадении секундной стрелки на изображении"""

    match_value: int
    """ Результат совпадения, чем больше значение, тем лучше результат """
    arrow_start: Point
    """ Координаты начала стрелки """
    angle_deg: float
    """ Угол поворота стрелки в градусах """


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


RESULT_IMAGE_REGULAR = re.compile(
    r'^(?P<error_status>[01]+)-(?P<error>\d+\.\d+)-(?P<detected_time>\d{2}:\d{2}:\d{2}\.\d{3})-'
    r'(?P<excepted_time>\d{2}:\d{2}:\d{2}\.\d{3})$'
)


@dataclass
class DetectTimeResult:
    success_detection: int
    error_sec: float
    detected_time: datetime
    excepted_time: datetime

    def to_str(self) -> str:
        """
        Формирует строку из результатов алгоритма определения времени по аналоговым часам

        :return: строку из результатов алгоритма определения времени
        """
        success_detection = 1 if self.success_detection else 0
        return (
            f'{success_detection}-{self.error_sec}-{self.detected_time.strftime("%H:%M:%S.%f")[:-3]}-'
            f'{self.excepted_time.strftime("%H:%M:%S.%f")[:-3]}'
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
