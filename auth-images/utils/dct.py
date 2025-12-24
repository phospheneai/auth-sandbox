import torch
import torch.nn.functional as F
import numpy as np
from scipy.fft import dct, idct

def dct2d(x):
    """
    Apply 2D Discrete Cosine Transform (DCT) to a batch of images.
    """
    return dct(dct(x, axis=-1, norm='ortho'), axis=-2, norm='ortho')

def idct2d(x):
    """
    Apply 2D DCT to a batch of images.
    """
    return idct(idct(x, axis=-1, norm='ortho'), axis=-2, norm='ortho')

def apply_dct_batch(images):
    """
    Apply 2D DCT to a batch of images.
    """

    batch_size, channels, height, width = images.shape
    dct_images = torch.zeros_like(images)

    for b in range(batch_size):
        for c in range(channels):
            img = images[b, c].cpu().numpy()
            dct_img = dct2d(img)
            dct_images[b, c] = torch.from_numpy(dct_img)

    return dct_images

def apply_idct_batch(dct_images):
    """
    Apply inverse 2D DCT to a batch of images of DCT coefficients.
    """
    batch_size, channels, height, width = dct_images.shape
    images = torch.zeros_like(dct_images)

    for b in range(batch_size):
        for c in range(channels):
            dct_img = dct_images[b, c].cpu().numpy()
            img = idct2d(dct_img)
            images[b, c] = torch.from_numpy(img)

    return images