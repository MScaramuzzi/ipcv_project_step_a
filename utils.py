import numpy as np
import matplotlib.pyplot as plt

def centroid(vertexes):
    vertexes = np.array(vertexes)  # Converti in array NumPy per operazioni efficienti
    return tuple(np.round(vertexes.mean(axis=0)).astype(int))  # Calcola la media lungo le colonne

def euclidian_distance(point_1, point_2):
    return np.linalg.norm(np.array(point_1) - np.array(point_2))  # Distanza euclidea con NumPy


def findmodel(i, check=False):
    models = {
        0: "Nesquik Cioccomilk",
        1: "ChocoKrave al latte",
        11: "ChocoKrave Nocciole",
        19: "Country crisp",
        24: "Fitness",
        25: "Coco Pops",
        26: "Nesquik Duo"
    }
    
    if i in models:
        if check:
            print(f"model {i}: {models[i]}")
        return i, models[i]
    else:
        print(f"⚠️ Attention: model {i} not found!") if check else None
        return None, None




def show_images(images, titles=None, cmap="gray"):
    """Mostra una lista di immagini con titoli opzionali."""
    for i, img in enumerate(images):
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
        plt.title(titles[i] if titles else f"Image {i}")
        plt.show()


def compute_reference_colors(models_rgb):
    """Calcola il colore medio della parte inferiore di ogni modello."""
    reference_colors = []
    for img in models_rgb:
        start_row = img.shape[0] // 3  # Prende la parte inferiore dell'immagine
        mean_color = np.mean(img[start_row:, :], axis=(0, 1))
        reference_colors.append(mean_color.astype(int).tolist())
    return reference_colors