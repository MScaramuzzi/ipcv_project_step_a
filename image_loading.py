import cv2
from typing import Tuple
import os
import numpy as np

from glob import glob


def load_single_model(model_file_name, model_dir="models/"):
    """Carica le immagini dei modelli in RGB e in scala di grigi."""
    read_image = cv2.imread(os.path.join(model_dir,model_file_name))
    img_query = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY) 
    return img_query




def load_model_images(model_filenames, model_dir="models/"):
    """Carica le immagini dei modelli in RGB e in scala di grigi."""
    model_images = [cv2.imread(os.path.join(model_dir, f)) for f in model_filenames]
    model_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in model_images]
    model_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in model_images]
    return model_rgb, model_gray


# ----------------- Funzioni per la fase di acquisizione modelli ----------------- #

def load_single_scene(scene_filename, scene_dir="scenes/step_A/"):
    """Carica un'immagine di scena e la converte in RGB e scala di grigi."""
    img_path = os.path.join(scene_dir, scene_filename + ".png")
    img = cv2.imread(img_path)
    img_train = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_train


# def load_scene(scene_filename, scene_dir="scenes/step_A/"):
#     """Carica un'immagine di scena e la converte in RGB e scala di grigi."""
#     img_path = os.path.join(scene_dir, scene_filename + ".png")
#     img = cv2.imread(img_path)
#     scene_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     scene_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return scene_rgb, scene_gray


def split_channels(model_rgb):
    """Estrae i tre canali (R, G, B) dalle immagini modello."""
    red_channels = [img[:, :, 0] for img in model_rgb]
    green_channels = [img[:, :, 1] for img in model_rgb]
    blue_channels = [img[:, :, 2] for img in model_rgb]
    return red_channels, green_channels, blue_channels


# ----------------- 1️⃣ Funzione per combinare i keypoints dei tre canali ----------------- #

def combine_keypoints(kp_model_1, kp_model_2, kp_model_3):
    """
    Combina i keypoints estratti dai tre canali RGB per ogni modello.

    Parameters:
        kp_model_1, kp_model_2, kp_model_3 (list): Liste di keypoints per ogni canale.
    
    Returns:
        list: Lista dei keypoints combinati per ogni modello.
    """
    return [kp1 + kp2 + kp3 for kp1, kp2, kp3 in zip(kp_model_1, kp_model_2, kp_model_3)]


# ----------------- Funzione per caricamento e preprocessamento scena ----------------- #


def split_scene_channels(scene_rgb):
    """Divide i tre canali della scena."""
    return scene_rgb[:, :, 0], scene_rgb[:, :, 1], scene_rgb[:, :, 2]




# def get_models_image(folder: str, model_list_idx: list[int]) -> Tuple[list,list]:
#     """From a given folder retrieve the corresponding model images 

#     Args:
#         folder (str): _description_
#         model_list_idx (list[int]): _description_

#     Returns:
#         _type_: _description_
#     """    
    
#     images = [cv2.imread(folder + str(model_idx)+".jpg") for model_idx in model_list_idx]
#     models = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images]
#     models_gray = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]

#     return models, models_gray

# def get_scenes_image(folder: str):
    # """From a given folder retrieve the corresponding model images

    # Args:
    #     folder (str): _description_

    # Returns:
    #     _type_: _description_
    # """    

    # images =[cv2.imread(img) for img in glob(f"{folder}*") ]
    # scenes = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images]
    # scenes_gray = [cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) for image in images]

    # return scenes, scenes_gray