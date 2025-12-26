import os
import yaml
import torch
import json
from PIL import Image
import torchvision.transforms as transforms

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def save_metrics(metrics, save_path):
    """Save metrics to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_image(image_path, transform=None):
    """Load and transform image"""
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    return image

def save_image(tensor, save_path):
    """Save tensor as image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if needed
    tensor = torch.clamp(tensor, 0, 1)
    
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    image.save(save_path)

def create_run_dir(base_dir="runs"):
    """Create unique run directory"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
