import cv2


def read_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"图片读取失败：{image_path}")

    return image


def convert_to_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def threshold_image(gray_image, threshold_value):
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image


def calculate_defect_area(binary_image):
    defect_area = cv2.countNonZero(binary_image)
    return defect_area


def find_defect_contours(binary_image):
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def draw_defect_boxes(image, contours, min_area):
    output_image = image.copy()

    defect_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                output_image,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )

            defect_count = defect_count + 1

    return output_image, defect_count