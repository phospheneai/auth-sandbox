import os, torch, sys
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import torch_dct as dct

sys.path.append('..')
from seed import seed
from utils.io import load_config, save_image

def apply_dct_batch(x):
    return dct.dct_2d(x, norm="ortho")

def apply_idct_batch(x):
    return dct.idct_2d(x, norm="ortho")

def radial_mask(h, w, cutoff, device):
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device),
        torch.linspace(0, 1, w, device=device),
        indexing="ij"
    )
    r = torch.sqrt(xx**2 + yy**2)
    return (r <= cutoff).float()[None, None, :, :]

def frequency_swap(config):
    seed(config['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(config['data']['swapped_dir'], exist_ok=True)

    real_dir = config['data']['real_dir']
    recon_dir = config['data']['recon_dir']

    image_files = [
        f for f in os.listdir(real_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
    ])

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Frequency Swapping"):
            real_img = Image.open(os.path.join(real_dir, img_file)).convert('RGB')
            recon_img = Image.open(os.path.join(recon_dir, img_file)).convert('RGB')

            real_tensor = transform(real_img).unsqueeze(0).to(device)
            recon_tensor = transform(recon_img).unsqueeze(0).to(device)

            # Apply proper DCT
            real_dct = apply_dct_batch(real_tensor)
            recon_dct = apply_dct_batch(recon_tensor)

            # Create radial masks
            _, _, h, w = real_dct.shape
            lf_mask = radial_mask(h, w, config['dct']['cutoff_ratio'], device)
            hf_mask = 1 - lf_mask

            # Frequency swap: real HF + recon LF
            swapped_dct = real_dct * hf_mask + recon_dct * lf_mask

            # Inverse DCT
            swapped_img = apply_idct_batch(swapped_dct)
            swapped_img = torch.clamp(swapped_img, 0, 1)

            save_image(
                swapped_img, 
                os.path.join(config['data']['swapped_dir'], f"swapped_{img_file}")
            )

if __name__ == "__main__":
    config = load_config("../config.yaml")
    frequency_swap(config)
