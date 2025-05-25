import numpy as np
from typing import Tuple
import cv2
from numpy.typing import NDArray


def compute_aligned_rectangle(corners: np.ndarray,
                              img_shape: Tuple[int, int]) -> Tuple[np.ndarray, int, int, np.ndarray]:
    """
    Compute an axis-aligned rectangle approximation from four corner points, ensuring the rectangle
    remains within the image boundaries. The function calculates the rectangle's dimensions,
    center, and adjusted corner points to provide a simplified representation of the region.

    Args:
        corners (np.ndarray): A 4x1x2 array representing the four corner points of a quadrilateral.
        img_shape (Tuple[int, int]): The shape of the image as (height, width), used to constrain
                                     the rectangle within valid image boundaries.

    Returns:
        rectangle (np.ndarray): A 4x1x2 array representing the axis-aligned rectangle's corner points.
        width (int): The width of the rectangle, computed as the horizontal distance between corners.
        height (int): The height of the rectangle, computed as the vertical distance between corners.
        center (np.ndarray): The center point of the rectangle as a 2D coordinate (x, y).
    """

    # Extract the 4 points from the input array
    top_left = corners[0][0]
    bottom_left = corners[1][0]
    bottom_right = corners[2][0]
    top_right = corners[3][0]

    # Calculate the top and bottom Y coordinates by averaging the Y values
    top_y = round((top_left[1] + top_right[1]) / 2)
    bottom_y = round((bottom_left[1] + bottom_right[1]) / 2)

    # Calculate the left and right X coordinates by averaging the X values
    left_x = round((top_left[0] + bottom_left[0]) / 2)
    right_x = round((top_right[0] + bottom_right[0]) / 2)

    # Ensure the rectangle stays within the image boundaries
    top_y = max(0, top_y)
    bottom_y = min(img_shape[0] - 1, bottom_y)
    left_x = max(0, left_x)
    right_x = min(img_shape[1] - 1, right_x)

    # Construct the rectangle using the axis-aligned corner points
    rectangle = np.array([
        [[left_x, top_y]],
        [[left_x, bottom_y]],
        [[right_x, bottom_y]],
        [[right_x, top_y]]
    ])

    # Compute width and height using Euclidean distance
    width = round(np.linalg.norm(rectangle[3][0] - rectangle[0][0]))   # Horizontal side
    height = round(np.linalg.norm(rectangle[1][0] - rectangle[0][0]))  # Vertical side

    # Compute the center point of the rectangle
    center = np.array([
        round((left_x + right_x) / 2),
        round((top_y + bottom_y) / 2)
    ])

    return rectangle, width, height, center


def draw_bounding_box(img_train: NDArray[np.uint8],
                    dst: NDArray[np.float32]) -> NDArray[np.uint8]:

    """
    Draws a bounding box on the given image using the provided destination points.

    Args:
        img_train (NDArray[np.uint8]): The image on which the bounding box will be drawn.
        dst (NDArray[np.float32]): The destination points for the bounding box.

    Returns:
        NDArray[np.uint8]: The image with the bounding box drawn.
    """

    return cv2.polylines(
        img_train,
        pts=[np.int32(dst)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=4,
        lineType=cv2.LINE_AA
    )



def draw_text_with_background(img: np.ndarray,
                            text: str,
                            center: np.ndarray,
                            font=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=3,
                            thickness=0.5,
                            bg_color=(255, 255, 255),
                            opacity=0.5,
                            text_color=(0, 0, 0)) -> np.ndarray:
    """
    Draw text at the given center with a semi-transparent background.
    The position of the background rectangle is adjusted such that the text is centered.

    Args:
        img (np.ndarray): Image on which to draw.
        text (str): Text to draw.
        center (np.ndarray): Center coordinate (x, y) where the text should be centered.
        font: OpenCV font type.
        font_scale (float): Scale factor that is multiplied by the font-specific base size.
        thickness (int): Thickness of the text strokes.
        bg_color (tuple): Background rectangle color.
        opacity (float): Opacity of the background rectangle (0 to 1).
        text_color (tuple): Color of the text.

    Returns:
        np.ndarray: Modified image with the text drawn.
    """

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate top-left corner of the text box so that the text is centered
    x = int(center[0] - text_width / 2)
    y = int(center[1] - text_height / 2)

    # Add a margin around the text
    margin = 5
    rect_top_left = (x - margin, y - margin)
    rect_bottom_right = (x + text_width + margin, y + text_height + margin)

    # Create an overlay for the background
    overlay = img.copy()
    cv2.rectangle(overlay, rect_top_left, rect_bottom_right, bg_color, -1)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # Put the text (OpenCV putText expects bottom-left coordinate for the text)
    cv2.putText(img, text, (x, y + text_height), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return img



def height_correction(rectangle, width, height, center, model_img, img_shape, threshold=1.15):
    import numpy as np

    model_h, model_w = model_img.shape[:2]
    model_aspect = model_h / model_w
    detected_aspect = height / width

    if detected_aspect > model_aspect * threshold:
        new_height = int(width * model_aspect)
        cx, cy = center
        img_h, img_w = img_shape[:2]

        # Compute top and bottom, keep center and width
        top_y = int(np.clip(cy - new_height // 2, 0, img_h-1))
        bottom_y = int(np.clip(cy + new_height // 2, 0, img_h-1))
        left_x = rectangle[0][0][0]
        right_x = rectangle[2][0][0]

        # Rebuild rectangle
        corrected_rectangle = np.array([
            [[left_x, top_y]],
            [[left_x, bottom_y]],
            [[right_x, bottom_y]],
            [[right_x, top_y]]
        ])
        return corrected_rectangle, width, (bottom_y-top_y), center
    else:
        return rectangle, width, height, center
