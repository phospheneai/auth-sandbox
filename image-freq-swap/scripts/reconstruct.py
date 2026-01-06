import os, torch, sys
from diffusers import AutoencoderKL
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import random
import numpy as np
import yaml

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_image(tensor, save_path):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = T.ToPILImage()
    image = to_pil(tensor)
    image.save(save_path)

def reconstruct_only(config):
    seed(config['seed'])
    device = config['vae']['device']

    vae = AutoencoderKL.from_pretrained(config['vae']['model_name']).to(device)
    vae.eval()

    vae_transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    os.makedirs(config['data']['recon_dir'], exist_ok=True)

    real_dir = config['data']['real_dir']
    all_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_files.sort()
    
    start_idx = all_files.index("n04229816_16592.JPEG")
    image_files = all_files[start_idx:]

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="VAE Reconstruction"):
            img_path = os.path.join(real_dir, img_file)
            image = Image.open(img_path).convert("RGB")
            
            orig_w, orig_h = image.size

            image_tensor = vae_transform(image).unsqueeze(0).to(device)
            latents = vae.encode(image_tensor).latent_dist.mode()
            recon = vae.decode(latents).sample
            recon = (recon + 1) / 2

            recon = F.interpolate(
                recon,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False
            )

            save_image(recon, os.path.join(config['data']['recon_dir'], img_file))

if __name__ == "__main__":
    config_path = os.path.join(parent_dir, "config.yaml")
    config = load_config(config_path)
    reconstruct_only(config)
