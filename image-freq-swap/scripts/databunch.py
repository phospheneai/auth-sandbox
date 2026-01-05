import os, random, torch, sys
from PIL import Image
import torchvision.transforms.functional as tvf
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.append('../utils')
from frequency_mix import FreqMix, PixelBlendMix
import random

class PairedAugmentation:
    """
    Applies the SAME spatial / photometric transforms
    to (real, mixed) images.
    """

    def __init__(self):
        self.size = 336
        self.transform = A.Compose(
            [
                # Ensure min size
                A.SmallestMaxSize(max_size=self.size, p=1.0),

                # Shared random crop
                A.RandomCrop(height=self.size, width=self.size, p=1.0),

                # JPEG artifacts
                A.ImageCompression(
                    quality_lower=50,
                    quality_upper=100,
                    p=1.0
                ),

                # Mild blur
                A.GaussianBlur(
                    blur_limit=(3, 3),
                    sigma_limit=(0.5, 1.0),
                    p=1.0
                ),

                # Flip
                A.HorizontalFlip(p=0.5),

                # ImageNet normalization
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),

                ToTensorV2(),
            ],
            additional_targets={
                "pair": "image"
            }
        )

    def __call__(self, img, pair_img):
        img = np.array(img)
        pair_img = np.array(pair_img)

        augmented = self.transform(image=img, pair=pair_img)

        return augmented["image"], augmented["pair"]


# =========================
# DATASET
# =========================
class DataPipeline(Dataset):
    """
    REAL + RECON dataset with on-the-fly mixing.

    Label convention:
        1 → FreqMix(real, recon)
        0 → PixelBlend(real, recon)
    """

    def __init__(
        self,
        real_dir,
        recon_dir,
        freq_ratios=(0.0, 0.8),
        pixel_ratios=(0.0, 0.8),
        seed=42,
    ):
        super().__init__()

        self.real_paths = sorted(Path(real_dir).glob("*"))
        self.recon_paths = sorted(Path(recon_dir).glob("*"))

        if len(self.real_paths) != len(self.recon_paths):
            raise RuntimeError(
                "❌ real_dir and recon_dir must contain aligned images"
            )

        self.seed = seed

        self.freq_mixer = FreqMix(
            ratios=list(freq_ratios),
            random_seed=None
        )

        self.pixel_mixer = PixelBlendMix(
            ratios=list(pixel_ratios),
            random_seed=None
        )

        self.transform = PairedAugmentation()

        print("[DataPipeline initialized]")
        print(f"  Samples       : {len(self.real_paths)}")
        print("  Mode          : Random FreqMix / PixelBlend")
        print("  Output size   : 224×224")
        print("  Augmentation  : Albumentations (paired)")

    def __len__(self):
        return len(self.real_paths)

    def __getitem__(self,i):
        real_img = Image.open(self.real_paths[i]).convert("RGB")
        recon_img = Image.open(self.recon_paths[i]).convert("RGB")
        if random.randint(0,1) == 0:
            mix_img = self.freq_mixer(real_img,recon_img)
        else:
            mix_img = self.pixel_mixer(real_img,recon_img)
        real, mixed = self.transform(real_img,mix_img)
        return real,mixed,0,1
        # return real_img,recon_i

def paired_collate_fn(batch):
    """
    batch: list of tuples (img, pair, 0, 1)
    Output: shuffled stacked tensors and labels
    """
    images = []
    labels = []

    for img, pair, lbl_img, lbl_pair in batch:
        # Append both image and pairimage
        images.append(img)
        labels.append(lbl_img)

        images.append(pair)
        labels.append(lbl_pair)

    # Stack into batch tensor
    images = torch.stack(images, dim=0)  # shape: [B, C, H, W]
    labels = torch.tensor(labels)

    # Shuffle consistently
    indices = list(range(len(labels)))
    random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

    return images, labels