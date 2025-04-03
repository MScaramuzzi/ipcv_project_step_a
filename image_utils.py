import cv2
from typing import Tuple
import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from numpy.typing import NDArray


def load_images(base_path: str, step_directory: str, image_indices: list[int] = [1, 2, 3, 4, 5]) -> dict[int, NDArray[np.uint8]]:
    """
    Loads images from a specified directory, processes them, and returns a dictionary
    mapping image indices to their corresponding processed images.
    Args:
        base_path (str): The base directory path where images are stored.
                         It can be 'models' or 'scenes'.
        step_directory (str): The subdirectory within the base path to look for images.
                              For example, 'step_A', 'step_B', or 'step_C'.
        image_indices (list, optional): A list of numerical indices to map to the loaded images.
                                         Defaults to [1, 2, 3, 4, 5].
    Returns:
        dict: A dictionary where keys are the provided image indices and values are the
              processed images. Images are loaded in RGB format. If the base path is 'models',
              the images are resized to (180, 250). Otherwise, they are returned in their
              original dimensions.
    Raises:
        AssertionError: If the number of images found in the directory does not match the
                        length of the `image_indices` list.
    Notes:
        - The function determines the file extension to look for based on the `base_path`
          and `step_directory` arguments. It uses '.jpg' for 'models' or 'scenes/step_C',
          and '.png' otherwise.
        - The function uses OpenCV to read and process images.
    """
    if (base_path == 'models') or (base_path == 'scenes' and step_directory == 'step_C'):
        end_path = '.jpg'
    else:
        end_path = '.png'

    current_path = os.path.join(base_path, step_directory)

    # Construct the pattern by adding the separator and the wildcard
    pattern = os.path.join(current_path, f'*{end_path}')
    images = glob(pattern)

    assert(len(image_indices) == len(images))

    # read the image and use the current numerical index to retrieve the correct img_reference number
    if base_path == 'scenes':
        images_dict = {image_indices[index]: cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
                    for index, image in enumerate(images)}
    else:
        images_dict = {image_indices[index]: cv2.resize(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), dsize=(180, 250)) #FIXME
                    for index, image in enumerate(images)}

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
    rows = (num_images + n_cols - 1) // n_cols  # Calcola il numero di righe necessarie

    plt.figure(figsize=(15, 4 * rows), dpi=100)
    for i, img in enumerate(list(images_dict.values())):
        plt.subplot(rows, n_cols, i + 1)
        plt.imshow(img)
        if title:
            plt.title(f'Image num. {list(images_dict.keys())[i]}')
            plt.suptitle(title,fontsize=15)
        # plt.axis('off')
    plt.show()


def plot_bbox(img_train_bounding: NDArray[np.uint8], model_id: int, scene_id: int) -> None:
    """
    Plots the bounding box of a detected model on the scene image.

    Args:
        img_train_bounding (NDArray[np.uint8]): The scene image with the bounding box drawn on it.
        model_id (int): The ID of the detected model.
        scene_id (int): The ID of the scene being analyzed.

    Returns:
        None: Displays the image with the bounding box using matplotlib.
    """

    plt.figure(figsize=(15, 8), dpi=100)
    plt.imshow(img_train_bounding, 'gray', vmin=0, vmax=255)
    plt.title(f'Drawing bounding box of model {model_id} for scene {scene_id}', fontsize=13)
    plt.show()

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
