import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Función para guardar imágenes generadas
def save_generated_images(images, path, prefix="generated"):
    os.makedirs(path, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, os.path.join(path, f"{prefix}_{i}.png"))


def calculate_ssim(img1, img2):
    # img1 e img2 deben estar en formato numpy, rango [0, 255] para imágenes en escala de grises
    img1 = img1.squeeze()  # Eliminar dimensiones innecesarias
    img2 = img2.squeeze()  # Eliminar dimensiones innecesarias

    ssim_value = ssim(img1, img2, data_range=img2.max() - img2.min())
    return ssim_value

def calculate_psnr(img1, img2):
    # img1 e img2 deben estar en formato numpy, rango [0, 255] para imágenes en escala de grises
    img1 = img1.squeeze()  # Eliminar dimensiones innecesarias
    img2 = img2.squeeze()  # Eliminar dimensiones innecesarias

    psnr_value = psnr(img1, img2, data_range=img2.max() - img2.min())
    return psnr_value
