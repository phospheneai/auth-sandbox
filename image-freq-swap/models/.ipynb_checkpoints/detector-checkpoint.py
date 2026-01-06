import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, CLIPVisionModel, AutoImageProcessor
import torchvision.models as models
import torchvision.transforms as T

class HFLFDetector(nn.Module):
    def __init__(self, backbone='dino', model_name=None, num_classes=2):
        super().__init__()
        self.backbone_type = backbone
        
        if backbone == 'dino':
            self.backbone = AutoModel.from_pretrained(
                model_name or 'facebook/dinov2-base'
            )
            
            self.feature_dim = self.backbone.config.hidden_size
            self.processor = AutoImageProcessor.from_pretrained(
                model_name or 'facebook/dinov2-base'
            )
            
        elif backbone == 'clip':
            self.backbone = CLIPVisionModel.from_pretrained(
                model_name or 'openai/clip-vit-base-patch32'
            )
            
            self .feature_dim = self.backbone.config.hidden_size
            self.processor = AutoImageProcessor.from_pretrained(
                model_name or 'openai/clip-vit-base-patch32'
            )
            
        else:
            self.backbone = models.resnet18(pretrained=True)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.processor = None  # No specific processor for ResNet
            
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        # x is already preprocessed tensor:
        if self.backbone_type in ['dino', 'clip']:
            # ViT expects specific preprocessing
            if self.backbone_type == 'dino':
                outputs = self.backbone(pixel_values=x)
                features = outputs.last_hidden_state[:, 0] # CLS token
                
            else:
                outputs = self.backbone(pixel_values=x)
                features = outputs.pooler_output
                
        else:
            features = self.backbone(x)
            
        return self.classifier(features)
    
    def get_preprocessing_transform(self):
        """ Returns the preprocessing transform for the backbone. """
        if self.processor is not None:
            # For DINO/CLiP, use the processor's preprocessing
            def transform(image_tensor):
                # Convert tensor to PIL for processor, then back to tensor
                if image_tensor.dim() == 4:
                    image_tensor = image_tensor.unsqueeze(0)
                    
                # processor expects PIL or Numpy 
                to_pil = T.ToPILImage()
                pil_img = to_pil(image_tensor)
                
                # process and return tensor
                processed = self.processor(pil_img, return_tensors="pt")
                return processed['pixel_values'].squeeze(0)
            return transform
        else:
            # For ResNet, use standard normalization Imagenet
            return T.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
def create_detector(config):
    return HFLFDetector(
        backbone=config['model']['backbone'],
        model_name=config['model'].get('model_name'),
        num_classes=config['model']['num_classes']
    )