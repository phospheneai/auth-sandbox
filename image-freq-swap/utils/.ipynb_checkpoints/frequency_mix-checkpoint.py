import torch
import torch.nn as nn
import numpy as np
import random
from scipy.fftpack import dct, idct
from PIL import Image

class FreqMix(nn.Module):
    def __init__(self, ratios=[0.0, 0.85], patch=-1, random_seed=None):
        super().__init__()
        self.ratios = ratios
        self.patch = patch
        self.random_seed = random_seed

    def apply_2d_dct(self, patch):
        return dct(dct(patch.T, norm="ortho").T, norm="ortho")

    def apply_2d_idct(self, patch):
        return idct(idct(patch.T, norm="ortho").T, norm="ortho")

    def process_patches(self, real_channel, fake_channel, blend_ratio):
        height, width = real_channel.shape
        mixed_channel = np.zeros_like(real_channel, dtype=np.float32)
        step_size = self.patch // 2 if self.patch >= 16 else self.patch
        normalization_map = np.zeros_like(real_channel, dtype=np.float32)

        for i in range(0, height, step_size):
            for j in range(0, width, step_size):
                end_i = min(i + self.patch, height)
                end_j = min(j + self.patch, width)
                real_p = real_channel[i:end_i, j:end_j].astype(np.float32)
                fake_p = fake_channel[i:end_i, j:end_j].astype(np.float32)
                curr_h, curr_w = real_p.shape
                
                if (curr_h, curr_w) != (self.patch, self.patch):
                    pad_h = self.patch - curr_h
                    pad_w = self.patch - curr_w
                    real_p = np.pad(real_p, ((0, pad_h), (0, pad_w)), mode="edge")
                    fake_p = np.pad(fake_p, ((0, pad_h), (0, pad_w)), mode="edge")

                real_dct = self.apply_2d_dct(real_p)
                fake_dct = self.apply_2d_dct(fake_p)
                mixed_dct = blend_ratio * real_dct + (1 - blend_ratio) * fake_dct
                mixed_p = self.apply_2d_idct(mixed_dct)
                mixed_p = mixed_p[:curr_h, :curr_w]
                
                mixed_channel[i:end_i, j:end_j] += mixed_p
                normalization_map[i:end_i, j:end_j] += 1.0

        normalization_map[normalization_map == 0] = 1.0
        mixed_channel = mixed_channel / normalization_map
        return np.round(np.clip(mixed_channel, 0, 255)).astype(np.uint8)

    @torch.no_grad()
    def forward(self, real_img, fake_img):


        if isinstance(real_img, torch.Tensor):
            if real_img.dim() == 4:
                real_img = real_img.squeeze(0)
            real_img = real_img.cpu().permute(1, 2, 0).numpy()
            real_img = (real_img * 255).astype(np.uint8)
            real_img = Image.fromarray(real_img)
            
        if isinstance(fake_img, torch.Tensor):
            if fake_img.dim() == 4:
                fake_img = fake_img.squeeze(0)
            fake_img = fake_img.cpu().permute(1, 2, 0).numpy()
            fake_img = (fake_img * 255).astype(np.uint8)
            fake_img = Image.fromarray(fake_img)

        if real_img.size != fake_img.size:
            fake_img = fake_img.resize(real_img.size, Image.LANCZOS)

        real_np = np.array(real_img.convert("RGB"))
        fake_np = np.array(fake_img.convert("RGB"))
        
        ratio = random.uniform(self.ratios[0], self.ratios[1])
        mixed_channels = []

        for c in range(3):
            if self.patch == -1:
                real_dct = self.apply_2d_dct(real_np[:, :, c].astype(np.float32))
                fake_dct = self.apply_2d_dct(fake_np[:, :, c].astype(np.float32))
                mixed_dct = ratio * real_dct + (1 - ratio) * fake_dct
                mixed_c = self.apply_2d_idct(mixed_dct)
                mixed_channels.append(np.round(np.clip(mixed_c, 0, 255)).astype(np.uint8))
            else:
                mixed_c = self.process_patches(real_np[:, :, c], fake_np[:, :, c], ratio)
                mixed_channels.append(mixed_c)

        return Image.fromarray(np.stack(mixed_channels, axis=2))

class PixelBlendMix(nn.Module):
    def __init__(self, ratios=(0.0, 0.85), random_seed=None):
        super().__init__()
        self.ratios = ratios
    @torch.no_grad()
    def forward(self, real_img, fake_img):

        if real_img.size != fake_img.size:
            fake_img = fake_img.resize(real_img.size, Image.LANCZOS)

        real_arr = np.array(real_img.convert("RGB")).astype(np.float32)
        fake_arr = np.array(fake_img.convert("RGB")).astype(np.float32)

        blend_factor = random.uniform(self.ratios[0], self.ratios[1])
        blended = blend_factor * fake_arr + (1 - blend_factor) * real_arr
        
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
