from pathlib import Path

from tabulate import tabulate

from test_clock_detection.data_types import DetectTimeResult


def _take_result_from_data(data_folder: Path) -> list[float]:
    """
    Парсит результаты тестирования из директории с изображениями.

    :param data_folder: путь до директории с результатами тестирования
    :return: список с результатами тестирования
    """

    test_results = []
    for image in data_folder.glob('*.bmp'):
        image_name = image.stem
        result = DetectTimeResult.from_str(image_name)
        test_results.append(result.error_sec)

    assert len(test_results) > 0, f'Не найден ни один результат в директории: {data_folder}'
    test_results.sort()

    return test_results


def _make_stats(error_accuracy: list[float], test_results: list[float]) -> dict[float, float]:
    """
    Создает словарь формата {предел погрешности: процент результатов удовлетворяющих погрешности}

    :param error_accuracy: пределы погрешностей
    :param test_results: список с результатами тестирования
    :return: словарь с пределами погрешностей и процентов результатов, входящих в эту погрешность
    """

    count_results_in_accuracy = {}
    for error in error_accuracy:
        count_results_in_accuracy[error] = 0

    for result in test_results:
        for error in error_accuracy:
            if result <= error:
                count_results_in_accuracy[error] += 1

    percent_results_in_accuracy = {}
    for error, count in count_results_in_accuracy.items():
        percent_in_accuracy = round(count / len(test_results) * 100, 2)
        percent_results_in_accuracy[error] = percent_in_accuracy

    return percent_results_in_accuracy


def _print_result_analysis(percent_results_in_accuracy: dict[float, float], data_size: int) -> None:
    """
    Выводит в консоль результаты анализа

    :param percent_results_in_accuracy: словарь с погрешностями и процентом вошедших в них
    результатов
    :return:
    """

    print(f'Размер выборки: {data_size}')
    table = []
    for accuracy, percent in percent_results_in_accuracy.items():
        error_count = data_size - round(data_size * percent / 100)
        table.append([f'{accuracy} c.', f'{percent} %', error_count])
    print(
        tabulate(
            table,
            headers=['Погрешность', 'Уложилось в погрешность', 'Количество ошибок'],
            tablefmt='github',
        )
    )


def create_report_of_test(path_to_data: Path, error_accuracy: list[float]) -> None:
    """
    Изменяет заданный csv файл, внося в него результаты пределы погрешностей и проценты
    результатов, удовлетворяющих погрешностям.

    :param path_to_data: путь до директории с результатами
    :param error_accuracy: список погрешностей
    :return:
    """

    test_results = _take_result_from_data(path_to_data)
    percent_results_in_accuracy = _make_stats(error_accuracy, test_results)
    data_size = len(test_results)
    _print_result_analysis(percent_results_in_accuracy, data_size)
