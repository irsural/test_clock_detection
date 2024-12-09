import cv2
from cv2.typing import MatLike, Point

from test_clock_detection.data_types import Template


def draw_templates(template: Template, image: MatLike) -> MatLike:
    """
    Находит контуры шаблонов с помощью cv2.findContours и метода cv2.CHAIN_APPROX_SIMPLE.
    Найденные контуры отрисовываются на изображении и возле них пишется значение угла поворота
    шаблона

    :param template: шаблон, который необходимо нарисовать на изображении
    :param image: изображение
    :return: изображение с выделенными контурами шаблонов и подписанными рядом их углами поворота
    """

    templ_img = template.image
    if len(templ_img.shape) == 3:
        templ_img = cv2.cvtColor(templ_img, cv2.COLOR_BGRA2GRAY)

    templ_height, templ_width = templ_img.shape[:2]

    mask = cv2.getRotationMatrix2D(
        ((templ_width - 1) / 2.0, (templ_height - 1) / 2.0), -template.angle_deg, 1
    )
    templ_copy = cv2.warpAffine(templ_img, mask, (templ_width, templ_height))

    contours, hierarchy = cv2.findContours(templ_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(
            image, contours, i, (0, 0, 255), 1, cv2.LINE_8, hierarchy, 0, offset=template.top_left
        )

    cv2.rectangle(
        image,
        (template.top_left[0], template.top_left[1] - 15),
        (template.top_left[0] + 50, template.top_left[1]),
        (0, 250, 255),
        -1,
    )

    cv2.putText(
        image,
        str(template.angle_deg),
        (template.top_left[0], template.top_left[1] - 4),
        cv2.QT_FONT_NORMAL,
        0.5,
        (0, 0, 0),
        1,
    )

    return image


def draw_line(image: MatLike, start_line: Point, end_line: Point) -> MatLike:
    """
    Рисует на изображении цветную красную линию

    :param image: изображение в формате RGBA
    :param start_line: координаты начала линии
    :param end_line: координаты конца линии
    :return:
    """
    cv2.line(image, start_line, end_line, (0, 0, 255, 255), 1)
    return image
