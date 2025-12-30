import os, random, torch, sys
from PIL import Image
import torchvision.transforms.functional as tvf
from pathlib import Path
sys.path.append('../utils')
from frequency_swap import *

class DataPipeline:
    def __init__(self,
                 recon_dir,
                 real_dir):
        
        self.recon_dir = Path(recon_dir)
        self.real_dir = Path(real_dir)
        self.dct_mixer = FreqMix(ratios=[0.0, 0.8])
        self.pixel_mixer = PixelBlendMix(ratios=[0.0, 0.8])

    def __iter__(self):
        recon_files = list(self.recon_dir.glob('*.png')) + list(self.recon_dir.glob('*.jpg'))
        random.shuffle(recon_files)

        for recon_path in recon_files:
            real_file = self.real_dir / recon_path.name
            if not real_file.exists():
                continue

            real_img = Image.open(real_file).convert('RGB')
            recon_img = Image.open(recon_path).convert('RGB')

            if random.random() < 0.5:
                mixed_img = self.dct_mixer(real_img, recon_img)
                label = 1

            else:
                mixed_img = self.pixel_mixer(real_img, recon_img)
                label = 0

            # Convert to tensor
            mixed_img = tvf.to_tensor(mixed_img)