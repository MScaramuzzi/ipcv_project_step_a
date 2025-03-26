import cv2
from typing import Tuple
import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob




def load_images(base_path, step_directory, image_indices=[1,2,3,4,5]):
    if (base_path == 'models') or (base_path == 'scenes' and step_directory == 'step_C'):
        end_path = '.jpg'
    else:
        end_path = '.png'

    current_path = os.path.join(base_path, step_directory)
    
    # Costruisco il pattern aggiungendo il separatore e il wildcard
    pattern = os.path.join(current_path, f'*{end_path}')
    images = glob(pattern)

    assert(len(image_indices) == len(images))

    # read the image and use the current numerical index to retrieve the correct img_reference number
    images_dict = {image_indices[index]: cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) 
                   for index,image in enumerate(images)}

    # print(f'Pattern usato: {pattern}')
    return images_dict


def show_images(images_dict, n_cols,title=''):

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