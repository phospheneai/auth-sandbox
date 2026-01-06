import math
import torch
import torch.nn as nn
import torch.hub

# Channel dimensions for different DINOv2 variants
CHANNELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# -------------------------------
# Base DINOv2 model wrapper
# -------------------------------
class DINOv2Model(nn.Module):
    def __init__(self, name, num_classes=1):
        super().__init__()
        print(f"Loading DINOv2 from hub: {name}")
        self.model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        self.fc = nn.Linear(CHANNELS[name], num_classes)

    def forward(self, x, return_feature=False):
        if hasattr(self.model, "forward_features"):
            features_dict = self.model.forward_features(x)
            features = features_dict["x_norm_clstoken"]
        else:
            features = self.model(x)
            if isinstance(features, dict):
                features = features.get(
                    "x_norm_clstoken", features.get("last_hidden_state", None)[:, 0]
                )

        if return_feature:
            return features, self.fc(features)
        return self.fc(features)

# -------------------------------
# LoRA components
# -------------------------------
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_out = torch.einsum("...d,rd->...r", x, self.lora_A)
        lora_out = torch.einsum("...r,or->...o", lora_out, self.lora_B)
        return lora_out * (self.alpha / self.rank)

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0, trainable_orig=False):
        super().__init__()
        self.original_layer = original_layer
        if not trainable_orig:
            for p in self.original_layer.parameters():
                p.requires_grad = False
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, rank, alpha)

    def forward(self, x):
        return self.original_layer(x) + self.lora(x)

    def __getattr__(self, name):
        if name in ["weight", "bias"]:
            return getattr(self.original_layer, name)
        return super().__getattr__(name)

def get_submodule(model, submodule_name):
    if not submodule_name:
        return model
    parts = submodule_name.split(".")
    m = model
    for part in parts:
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m

def apply_lora_to_linear_layers(model, rank=4, alpha=1.0, target_modules=None, trainable_orig=False):
    for name, module in model.named_modules():
        if target_modules and not any(t in name for t in target_modules):
            continue
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[1] if "." in name else name
            parent = model if parent_name == "" else get_submodule(model, parent_name)
            setattr(parent, child_name, LoRALinear(module, rank, alpha, trainable_orig))
    return model

def get_lora_params(model):
    params = []
    for _, module in model.named_modules():
        if isinstance(module, LoRALayer):
            params.extend(module.parameters())
        elif isinstance(module, LoRALinear):
            params.extend(module.lora.parameters())
    return params

# -------------------------------
# DINOv2 + LoRA wrapper
# -------------------------------
class DINOv2ModelWithLoRA(nn.Module):
    def __init__(self, name="dinov2_vitb14", num_classes=2, lora_rank=8, lora_alpha=1.0,
                 lora_targets=None):
        super().__init__()
        self.base_model = DINOv2Model(name, num_classes)
        if lora_targets is None:
            lora_targets = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
        print(f"Adding LoRA to DINOv2 (rank={lora_rank}, alpha={lora_alpha})")
        print(f"LoRA target modules: {lora_targets}")
        self.base_model.model = apply_lora_to_linear_layers(
            self.base_model.model,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_targets,
            trainable_orig=False,
        )
        self._get_lora_params = lambda: get_lora_params(self.base_model.model)

    def get_trainable_params(self):
        return list(self._get_lora_params()) + list(self.base_model.fc.parameters())

    def forward(self, x, return_feature=False):
        return self.base_model(x, return_feature=return_feature)