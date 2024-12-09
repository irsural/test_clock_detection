from pathlib import Path

import cv2

from test_clock_detection.algorithm_debugger import Debugger, DummyDebugger
from test_clock_detection.utils import ClockTime, Template, Line


def detect_time(image_path: Path, debug_mode: None | Debugger = None) -> ClockTime:
    """
    Скрипт определения времени на часах.
    Ниже приведены примеры как пользоваться дебагером, чтобы сохранять промежуточные результаты
    работы алгоритма.

    :param image_path: путь к изображению часов
    :param debug_mode: режим отладки
    :return: время на часах в формате чч:мм:сс.мс
    """
    debugger = debug_mode if debug_mode is not None else DummyDebugger()

    image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)

    # Сохранение изображения
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    debugger.save_image('Серое изображение', image_gray)

    # Создание изображения центра картинки
    top_left_image_center = (image.shape[1] // 4, image.shape[0] // 4)
    bottom_right_image_center = (image.shape[1] // 4 + 300, image.shape[0] // 4 + 300)
    center_image = image_gray[
        top_left_image_center[1] : bottom_right_image_center[1],
        top_left_image_center[0] : bottom_right_image_center[0],
    ]
    # Сохранение изображения центра
    debugger.save_image('Центр изображения', center_image)

    # Обработка для лучшего выделения контуров
    center_templ_image_canny = cv2.Canny(center_image, 100, 200)
    # Создание экземпляра шаблона для отрисовки его контуров
    center_templ = Template(
        name='Центр изображения',
        image=center_templ_image_canny,
        top_left=top_left_image_center,
        angle_deg=0,
    )
    # Отрисовка контуров центра картинки на оригинальном изображении
    debugger.save_image_with_contours(
        'Контур центра на изображении', image, 0, [center_templ], None
    )

    # Создания экземпляров линий
    image_center = (315, 250)
    line_1 = Line('Цветная линяя', image_center, 90, 200)
    line_2 = Line('Цветная линяя', image_center, 210, 200)
    # Отрисовка линий на оригинальном изображении
    debugger.save_image_with_contours(
        'Линия из центра изображения', image, 0, None, [line_1, line_2]
    )

    # Сохранение угла поворота часов
    clock_angle = 0
    debugger.save_device_angle(clock_angle)

    # Сохранение результата алгоритма определения времени
    result_time = ClockTime(hours=0, minutes=0, seconds=0, ms=0)
    return result_time
