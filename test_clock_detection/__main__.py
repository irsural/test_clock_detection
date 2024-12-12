import shutil
import os
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

from test_clock_detection.algorithm_debugger import AlgorithmDebugger
from test_clock_detection.detect_time import detect_time
from test_clock_detection.result_analysis import create_report_of_test
from test_clock_detection.utils import check_result
from test_clock_detection.data_types import DetectTimeResult
from pathlib import Path
from datetime import datetime
from test_clock_detection.const import (
    FAIL_DELTA_THRESHOLD_SECONDS,
    CALCULATED_ERRORS,
    PHOTO_EXTENSION
)


def _run_test_image(
    root_folder: Path,
    image_path: Path,
    folder_for_results: Path,
    debug_folder: Path,
    fail_threshold_seconds: float
) -> None:
    """
    Запускает тестирование алгоритма определения времени для 1 изображения. Сохраняет результат в
    директорию *files/Результат/Результаты*.
    Выводит в консоль имя файла и получившуюся погрешность при определении времени алгоритмом.

    :param root_folder: корневая папка проекта
    :param image_path: путь до изображения
    :param folder_for_results: путь для сохранения промежуточных этапов алгоритма только для тестируемого изображения
    :param debug_folder: путь до общей папки для сохранения итогового результата
    :param fail_threshold_seconds: максимальное отклонение от реального значения, после которого
      определение времени считается неудачным. Задается в секундах
    :return:
    """

    debug_folder_for_image = debug_folder / image_path.stem
    debug_folder_for_image.mkdir(parents=True, exist_ok=True)

    debugger = AlgorithmDebugger(debug_folder_for_image)
    result_time = detect_time(root_folder, image_path, debugger)

    result_time_dt = datetime.strptime(str(result_time), '%H:%M:%S.%f')
    excepted_time_24h = datetime.strptime(image_path.stem, '%H:%M:%S.%f')
    excepted_time_dt = datetime.strptime(excepted_time_24h.strftime('%I:%M:%S.%f'), '%I:%M:%S.%f')

    delta_sec, success_detection = check_result(
        excepted_time_dt, result_time_dt, fail_threshold_seconds
    )

    result = DetectTimeResult(
        success_detection, delta_sec, result_time_dt, excepted_time_dt
    ).to_str()

    result_algorithm_image_path = debugger.get_image_path('Линия из центра изображения')
    result_test_image_path = Path(f'{folder_for_results}/{result}.{PHOTO_EXTENSION}')
    shutil.copy(result_algorithm_image_path, result_test_image_path)
    print(f'{image_path.stem} : погрешность - {delta_sec}')


def run_tests(root_folder: Path) -> None:
    """
    Запускает тестирование алгоритма определения времени по всем изображения, которые находятся в
    указанной директории.
    Результаты различных этапов алгоритма каждого изображения находятся в папке:
    *files/Результат/Имя тестируемого изображения*
    Итоговый результат алгоритма по всех изображениям находится в папке: *files/Результат/Результаты*

    Формат имени файла с итоговым результатом через дефис: уложилась ли погрешность в максимальную
    погрешность, погрешность алгоритма, определенное алгоритмом время, действительное время на
    изображении, определенный алгоритмом угол поворота циферблата, действительный угол поворота
    циферблата.

    В результате тестирования в консоли будет выведена таблица, содержащая столбцы: погрешность,
    процент изображений, уложившихся в погрешность и количество изображений, не уложившихся в
    погрешность
    """

    data_folder = root_folder / 'files'
    input_photos_folder = data_folder / 'Изображения'
    results_by_steps_folder = data_folder / 'Результаты' / 'По шагам'
    final_results_folder = data_folder / 'Результаты' / 'Окончательные'

    results_by_steps_folder.mkdir(parents=True, exist_ok=True)
    final_results_folder.mkdir(parents=True, exist_ok=True)

    args_list = []
    for image_path in input_photos_folder.glob(f'*.{PHOTO_EXTENSION}'):
        args_list.append((
            root_folder,
            image_path,
            final_results_folder,
            results_by_steps_folder,
            FAIL_DELTA_THRESHOLD_SECONDS
        ))

    assert len(args_list) > 0, (
        f'В папке {input_photos_folder} нет изображений с расширением .{PHOTO_EXTENSION}'
    )

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(_run_test_image, *zip(*args_list, strict=False))

    create_report_of_test(final_results_folder, CALCULATED_ERRORS)


def main() -> None:
    repo_root = Path(os.path.abspath(__file__)).parent.parent
    run_tests(repo_root)


if __name__ == '__main__':
    main()
