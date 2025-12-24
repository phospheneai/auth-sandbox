import torch
import numpy as np

def create_circular_mask(height, width, cutoff_ratio):
    """
    Create a circular mask for frequency domain filtering.
    """
    center_h, center_w = height // 2, width // 2
    radius = min(center_h, center_w) * cutoff_ratio  # Fixed this line

    y, x = np.ogrid[:height, :width]
    mask = (x - center_w) ** 2 + (y - center_h) ** 2 <= radius ** 2

    return torch.from_numpy(mask.astype(np.float32))

def create_rectangular_mask(height, width, cutoff_ratio):
    """Create rectangular mask for LF/HF separation"""
    h_cut = int(height * cutoff_ratio)
    w_cut = int(width * cutoff_ratio)
    
    mask = torch.zeros(height, width)
    mask[:h_cut, :w_cut] = 1.0
    
    return mask

def get_mask(mask_type, height, width, cutoff_ratio):
    """Get mask based on type"""
    if mask_type == "circular":
        return create_circular_mask(height, width, cutoff_ratio)
    elif mask_type == "rectangular":
        return create_rectangular_mask(height, width, cutoff_ratio)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
