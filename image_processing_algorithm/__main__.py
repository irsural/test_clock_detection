import shutil
import os
from concurrent.futures.thread import ThreadPoolExecutor
import multiprocessing

from image_processing_algorithm.algorithm_debugger import AlgorithmDebugger
from image_processing_algorithm.detect_time import detect_time
from image_processing_algorithm.result_analysis import create_report_of_test
from image_processing_algorithm.utils import (
    check_result,
    DetectTimeResult,
)
from pathlib import Path
from datetime import datetime


def _run_test_image(
    image_path: Path, folder_for_results: Path, debug_folder: Path, accuracy_sec: int
) -> None:
    """
    Запускает тестирование алгоритма определения времени для 1 изображения. Сохраняет результат в
    директорию *files/Результат/Результаты*.
    Выводит в консоль имя файла и получившуюся погрешность при определении времени алгоритмом.

    :param image_path: путь до изображения
    :param folder_for_results: путь для сохранения промежуточных этапов алгоритма только для тестируемого изображения
    :param debug_folder: путь до общей папки для сохранения итогового результата
    :param accuracy_sec: предел максимальной допустимой погрешности
    :return:
    """

    debug_folder_for_image = debug_folder / image_path.stem
    debug_folder_for_image.mkdir(parents=True, exist_ok=True)

    debugger = AlgorithmDebugger(debug_folder_for_image)

    result_time = detect_time(image_path, debugger)
    result_time_dt = datetime.strptime(result_time.__str__(), '%H:%M:%S.%f')
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


def run_tests(path_to_images_dir: Path, debug_folder: Path) -> None:
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

    :param path_to_images_dir: путь до директории с изображениями с расширением .bmp
    :param debug_folder: путь до директории для сохранения результатов
    :return:
    """

    max_accuracy_sec = 1
    args_list = []
    debug_folder_for_results = debug_folder / 'Результаты'
    debug_folder_for_results.mkdir(parents=True, exist_ok=True)
    for image_path in path_to_images_dir.glob('*.bmp'):
        args_list.append((image_path, debug_folder_for_results, debug_folder, max_accuracy_sec))
        _run_test_image(image_path, debug_folder_for_results, debug_folder, max_accuracy_sec)

    assert len(args_list) > 0, 'В директории нет изображений с расширением .bmp'

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        executor.map(_run_test_image, *zip(*args_list, strict=False))

    list_accuracy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    create_report_of_test(debug_folder_for_results, list_accuracy)


def main() -> None:
    repo_root = Path(os.path.abspath(__file__)).parent.parent
    path_to_images_dir = repo_root / 'files' / 'Изображения'
    debug_folder = repo_root / 'files' / 'Результаты'
    run_tests(path_to_images_dir, debug_folder)


if __name__ == '__main__':
    main()
