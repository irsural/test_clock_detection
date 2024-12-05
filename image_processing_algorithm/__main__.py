import shutil
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
from image_processing_algorithm.algorithm_debugger import Debugger, DummyDebugger, AlgorithmDebugger
from image_processing_algorithm.utils import Template, Line, ClockTime, check_result, \
    DetectTimeResult
from pathlib import Path
from datetime import datetime


def detect_time(image_path: Path, debug_mode: None | Debugger = None) -> ClockTime:
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

    result_time = ClockTime(hours=0, minutes=0, seconds=0, ms=0)
    return result_time


def run_test(image_path: Path, folder_for_results: Path, debug_folder: Path, accuracy_sec: int) -> None:

    debug_folder_for_image = debug_folder / image_path.stem
    debug_folder_for_image.mkdir(parents=True, exist_ok=True)

    debugger = AlgorithmDebugger(debug_folder_for_image)

    result_time = detect_time(image_path, debugger)
    result_time_dt = datetime.strptime(result_time.__str__(), '%I:%M:%S.%f')
    excepted_time_24h = datetime.strptime(image_path.stem, '%H:%M:%S.%f')
    excepted_time_dt = datetime.strptime(excepted_time_24h.strftime('%I:%M:%S.%f'), '%I:%M:%S.%f')

    delta_sec, error_status = check_result(excepted_time_dt, result_time_dt, accuracy_sec)

    result = DetectTimeResult(
        error_status, delta_sec, result_time_dt, excepted_time_dt, debugger.get_device_angle(), 0
    ).to_str()

    result_algorithm_image_path = debugger.get_image_path('Контур центра на изображении')
    result_test_image_path = Path(f'{folder_for_results}/{result}.bmp')
    shutil.copy(result_algorithm_image_path, result_test_image_path)
    print(f'{image_path.stem} : погрешность - {delta_sec}')  # noqa: T201


def main() -> None:
    path_to_images_dir = Path('../files/Изображения')
    debug_folder = Path('../files/Результат')

    args_list = []
    debug_folder_for_results = debug_folder / "Результаты"
    debug_folder_for_results.mkdir(parents=True, exist_ok=True)
    for image_path in path_to_images_dir.glob('*.bmp'):
        args_list.append((image_path, debug_folder_for_results, debug_folder))

    assert len(args_list) > 0, 'В директории нет изображений с расширением .bmp'

    jobs = 1
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        executor.map(run_test, *zip(*args_list, strict=False))


if __name__ == '__main__':
    main()
