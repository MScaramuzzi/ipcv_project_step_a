from image_loading import get_scenes_image
import numpy as np
import matplotlib.pyplot as plt


def plot_color_bg_img(scenes_folder: str, scene_letter: str) -> None:
    """_summary_

    Args:None if scene_letter
        scenes (list): _description_
        scene_letter (str): _description_

    Returns:
        _type_: _description_
    """    
    scenes, scenes_grey = get_scenes_image(scenes_folder, scene_letter)
    random_img_idx = np.random.randint(0, len(scenes))
    random_img_col, random_img_bg = scenes[random_img_idx], scenes_grey[random_img_idx]
    
    plt.imshow(random_img_col, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(random_img_bg, cmap='gray', vmin=0, vmax=255)
    plt.show()
    
    return None