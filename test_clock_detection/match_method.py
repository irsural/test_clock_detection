from cv2.typing import MatLike, Point

from test_clock_detection.data_types import MatchResultLine
from test_clock_detection.utils import polar_to_cartesian


def find_line(
    src_image: MatLike,
    image_center: Point,
    angle_step_deg: float,
    min_len_line_pix: int,
    max_len_line_pix: int,
    color: tuple[int, int, int, int],
) -> list[MatchResultLine]:
    """
    Проверяет пиксели на белый цвет вокруг заданного центра. В месте с наибольшим количеством
    заданного цвета находится секундная стрелка

    :param src_image: черно-белое изображение циферблата
    :param image_center: координаты центра циферблата
    :param angle_step_deg: шаг поиска стрелки
    :param max_len_line_pix: максимальная длина искомой линии
    :param min_len_line_pix: минимальная длина искомой линии
    :param color: цвет в формате RGBA
    :return: параметры найденной секундной стрелки
    """
    match_result = []

    max_steps_angle = int(360 // angle_step_deg)
    for theta in np.linspace(start=0, stop=360, num=max_steps_angle):
        color_pixels = 0
        for radius in range(min_len_line_pix, max_len_line_pix):
            x, y = polar_to_cartesian(theta, radius, image_center, 90)

            try:
                if all(src_image[y][x] == color):
                    color_pixels += 1
            except IndexError:
                continue

        match_result.append(
            MatchResultLine(match_value=color_pixels, angle_deg=theta, arrow_start=image_center)
        )

    return match_result
