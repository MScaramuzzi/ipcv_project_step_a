import cv2
from typing import Tuple

from glob import glob


def get_models_image(folder: str, model_list_idx: list[int]) -> Tuple[list,list]:
    """From a given folder retrieve the corresponding model images 

    Args:
        folder (str): _description_
        model_list_idx (list[int]): _description_

    Returns:
        _type_: _description_
    """    
    
    images = [cv2.imread(folder + str(model_idx)+".jpg") for model_idx in model_list_idx]
    models = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images]
    models_gray = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]

    return models, models_gray

def get_scenes_image(folder: str):
    """From a given folder retrieve the corresponding model images

    Args:
        folder (str): _description_

    Returns:
        _type_: _description_
    """    

    images =[cv2.imread(img) for img in glob(f"{folder}*") ]
    scenes = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images]
    scenes_gray = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]

    return scenes, scenes_gray