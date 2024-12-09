from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path

import cv2
from cv2.typing import MatLike

from test_clock_detection.draw_image import draw_line, draw_templates
from test_clock_detection.utils import polar_to_cartesian
from test_clock_detection.data_types import Template, Line
from test_clock_detection.const import PHOTO_EXTENSION


class Debugger(ABC):
    """
    Абстрактный класс для отладчиков
    """

    @abstractmethod
    def save_image(self, image_name: str, image: MatLike) -> None:
        pass

    @abstractmethod
    def save_image_with_contours(
        self,
        image_name: str,
        image: MatLike,
        templates: list[Template] | None = None,
        lines: list[Line] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def get_debug_folder(self) -> Path:
        pass


class AlgorithmDebugger(Debugger):
    """
    Класс для отладки алгоритма
    """

    def __init__(self, debug_folder: Path) -> None:
        self.debug_folder = debug_folder
        """Путь до отладочной директории"""
        self.count_files_in_folder = 0
        """Количество сохраненных изображений в директории отладки"""
        self.image_names: dict[str, str] = {}
        """Словарь с именем изображения и его номером"""

    def save_image(self, image_name: str, image: MatLike) -> None:
        """
        Сохранение изображения с заданным именем в директорию для отладки

        :param image: изображение
        :param image_name: имя изображения
        :return:
        """
        assert image_name not in self.image_names, f'Файл {image_name} уже был сохранен'
        image_path = self._make_image_path(image_name)
        cv2.imwrite(image_path.as_posix(), image)
        self.count_files_in_folder += 1

    def save_image_with_contours(
        self,
        image_name: str,
        image: MatLike,
        templates: list[Template] | None = None,
        lines: list[Line] | None = None,
    ) -> None:
        """
        Сохраняет изображение с выделенными контурами шаблонов и линий

        :param image: изображение
        :param templates: список шаблонов
        :param lines:
        :param image_name: имя изображения
        :return: изображение
        """
        assert image_name not in self.image_names, f'Файл {image_name} уже был сохранен'
        result_image = image.copy()
        if templates is not None:
            result_image = self._draw_templates(result_image, templates)
        if lines is not None:
            result_image = self._draw_lines(result_image, lines)
        image_path = self._make_image_path(image_name)
        cv2.imwrite(image_path.as_posix(), result_image)
        self.count_files_in_folder += 1

    def get_debug_folder(self) -> Path:
        return self.debug_folder

    def _make_image_path(self, image_name: str) -> Path:
        file_name = f'{self.count_files_in_folder}. {image_name}'
        self.image_names[image_name] = file_name
        return self.debug_folder / f'{file_name}.{PHOTO_EXTENSION}'

    def get_image_path(self, image_name: str) -> Path:
        file_name = self.image_names[image_name]
        return self.debug_folder / f'{file_name}.{PHOTO_EXTENSION}'

    @staticmethod
    def _draw_templates(image: MatLike, templates: list[Template]) -> MatLike:
        image_copy = image.copy()
        if len(image_copy.shape) < 3:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        for template in templates:
            draw_templates(template, image_copy)
        return image_copy

    @staticmethod
    def _draw_lines(image: MatLike, lines: list[Line]) -> MatLike:
        image_copy = image.copy()
        if len(image_copy.shape) < 3:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        for line in lines:
            end_line_x, end_line_y = polar_to_cartesian(
                line.angle_deg, line.len_line, line.line_start, 90
            )
            draw_line(image_copy, line.line_start, (end_line_x, end_line_y))
        return image_copy


class DummyDebugger(Debugger):
    """
    Класс-затычка, когда не требуется отладка
    """

    def save_image(self, image_name: str, image: MatLike) -> None:
        pass

    def save_image_with_contours(
        self,
        image_name: str,
        image: MatLike,
        templates: list[Template] | None = None,
        lines: list[Line] | None = None,
    ) -> None:
        pass

    def get_debug_folder(self) -> Path:
        return Path('')
