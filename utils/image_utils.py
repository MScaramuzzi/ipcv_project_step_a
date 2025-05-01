import cv2
from typing import Tuple
import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from numpy.typing import NDArray


def load_images(
    base_path: str,
    step_directory: str,
    image_indices: list[int] = [1, 2, 3, 4, 5],
    is_resized: bool = False,
    target_model_dims: tuple[int,int] = (180, 250),
) -> dict[int, NDArray[np.uint8]]:
    """
    Load images from disk. If base_path is 'models', resize images to a standard height
    either passed manually or calculated automatically from the dataset.

    Args:
        base_path (str): 'models' or 'scenes'
        step_directory (str): Subdirectory path inside the base folder
        image_indices (list[int]): List of indices to assign to each image
        target_model_height (int): Height to which model images should be resized

    Returns:
        dict[int, np.ndarray]: Dictionary mapping index to RGB image
    """

    if (base_path == 'models') or (base_path == 'scenes' and step_directory == 'step_C'):
        end_path = '.jpg'
    else:
        end_path = '.png'

    current_path = os.path.join(base_path, step_directory)
    pattern = os.path.join(current_path, f'*{end_path}')
    images = glob(pattern)

    assert len(image_indices) == len(images), "Mismatch between image count and indices"

    images_dict = {}

    for idx, image_path in enumerate(images):
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_height, original_width = img_rgb.shape[:2]

        if base_path == 'models' and is_resized == True:
            # Print shape if the image is a model
            print(f'model: {image_indices[idx]} | {original_height = } | {original_width = }')

            # Resize model images to fixed or auto height
            img_rgb = cv2.resize(img_rgb, target_model_dims)

        # Save the model in the dictionary at the corresponding index
        images_dict[image_indices[idx]] = img_rgb

    return images_dict


def show_images(images_dict: dict, n_cols: int, title: str = '') -> None:
    """
    Displays a dictionary of images in a grid format using Matplotlib.
    Args:
        images_dict (dict): A dictionary where keys are image identifiers (e.g., names or indices)
                            and values are image data (e.g., NumPy arrays or PIL images).
        n_cols (int): The number of columns in the grid layout.
        title (str, optional): A title for the entire grid of images. Defaults to an empty string.
    Returns:
        None: This function does not return anything. It displays the images using Matplotlib.
    """

    num_images = len(images_dict.keys())
    rows = (num_images + n_cols - 1) // n_cols  # Compute the required number of rows

    plt.figure(figsize=(15, 4 * rows), dpi=100)
    for i, img in enumerate(list(images_dict.values())):
        plt.subplot(rows, n_cols, i + 1)
        plt.imshow(img)
        if title:
            plt.title(f'Image num. {list(images_dict.keys())[i]}')
            plt.suptitle(title,fontsize=15)
        # plt.axis('off')
    plt.show()


# def plot_bbox(img_train_bounding: NDArray[np.uint8], model_id: int, scene_id: int) -> None:
#     """
#     Plots the bounding box of a detected model on the scene image.

#     Args:
#         img_train_bounding (NDArray[np.uint8]): The scene image with the bounding box drawn on it.
#         model_id (int): The ID of the detected model.
#         scene_id (int): The ID of the scene being analyzed.

#     Returns:
#         None: Displays the image with the bounding box using matplotlib.
#     """

#     plt.figure(figsize=(15, 8), dpi=100)
#     plt.imshow(img_train_bounding, 'gray', vmin=0, vmax=255)
#     plt.title(f'Drawing bounding box of model {model_id} for scene {scene_id}', fontsize=13)
#     plt.show()

def draw_bounding_box(img_train: NDArray[np.uint8], dst: NDArray[np.float32]) -> NDArray[np.uint8]:
    """
    Draws a bounding box on the given image using the provided destination points.

    Args:
        img_train (NDArray[np.uint8]): The image on which the bounding box will be drawn.
        dst (NDArray[np.float32]): The destination points for the bounding box.

    Returns:
        NDArray[np.uint8]: The image with the bounding box drawn.
    """
    return cv2.polylines(
        img_train,  # Ensure the original image is not modified
        pts=[np.int32(dst)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=4,
        lineType=cv2.LINE_AA
    )
