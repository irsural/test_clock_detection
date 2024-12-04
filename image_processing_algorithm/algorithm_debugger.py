from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path

import cv2
from cv2.typing import MatLike

from image_processing_algorithm.draw_image import draw_line, draw_templates
from image_processing_algorithm.utils import Line, Template, polar_to_cartesian


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
        clock_angle_shift: float,
        templates: list[Template] | None = None,
        lines: list[Line] | None = None,
    ) -> None:
        pass

    @abstractmethod
    def save_device_angle(self, device_angle: float) -> None:
        pass

    @abstractmethod
    def get_device_angle(self) -> float:
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
        self.device_angle: float = field(default_factory=float)
        """Угол поворота прибора"""
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
        assert image_name not in self.image_names, f"Файл {image_name} уже был сохранен"
        image_path = self._make_image_path(image_name)
        cv2.imwrite(image_path.as_posix(), image)
        self.count_files_in_folder += 1

    def save_image_with_contours(
        self,
        image_name: str,
        image: MatLike,
        device_angle_shift: float,
        templates: list[Template] | None = None,
        lines: list[Line] | None = None,
    ) -> None:
        """
        Сохраняет изображение с выделенными контурами шаблонов и линий

        :param image: изображение
        :param templates: список шаблонов
        :param device_angle_shift: угол поворота прибора
        :param lines:
        :param image_name: имя изображения
        :return: изображение
        """
        assert image_name not in self.image_names, f"Файл {image_name} уже был сохранен"
        result_image = image.copy()
        if templates is not None:
            result_image = self._draw_templates(
                result_image, templates, device_angle_shift
            )
        if lines is not None:
            result_image = self._draw_lines(result_image, lines, device_angle_shift)
        image_path = self._make_image_path(image_name)
        cv2.imwrite(image_path.as_posix(), result_image)
        self.count_files_in_folder += 1

    def save_device_angle(self, device_angle: float) -> None:
        self.device_angle = device_angle

    def get_device_angle(self) -> float:
        try:
            float(self.device_angle)
        except ValueError:
            raise ValueError(
                self.device_angle, "Значение угла прибора не является числом"
            ) from None
        return self.device_angle

    def get_debug_folder(self) -> Path:
        return self.debug_folder

    def _make_image_path(self, image_name: str) -> Path:
        file_name = f"{self.count_files_in_folder}. {image_name}"
        self.image_names[image_name] = file_name
        return self.debug_folder / f"{file_name}.bmp"

    def get_image_path(self, image_name: str) -> Path:
        file_name = self.image_names[image_name]
        return self.debug_folder / f"{file_name}.bmp"

    @staticmethod
    def _draw_templates(
        image: MatLike, templates: list[Template], device_angle_shift: float
    ) -> MatLike:
        image_copy = image.copy()
        if len(image_copy.shape) < 3:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        for template in templates:
            template.angle_deg += device_angle_shift
            draw_templates(template, image_copy)
            template.angle_deg -= device_angle_shift
        return image_copy

    @staticmethod
    def _draw_lines(
        image: MatLike, lines: list[Line], device_angle_shift: float
    ) -> MatLike:
        image_copy = image.copy()
        if len(image_copy.shape) < 3:
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        for line in lines:
            line.angle_deg += device_angle_shift
            end_line_x, end_line_y = polar_to_cartesian(
                line.angle_deg, line.len_line, line.line_start, 90
            )
            draw_line(image_copy, line.line_start, (end_line_x, end_line_y))
            line.angle_deg -= device_angle_shift
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
        clock_angle_shift: float,
        templates: list[Template] | None = None,
        lines: list[Line] | None = None,
    ) -> None:
        pass

    def save_device_angle(self, device_angle: float) -> None:
        pass

    def get_device_angle(self) -> float:
        return 0

    def get_debug_folder(self) -> Path:
        return Path("")
