import os, yaml, random, torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from seed import seed
from models.detector import HFLFDetector

# Dataset Class
class FreqSwapDataset(Dataset):
    def __init__(self, real_paths, swapped_paths, transform=None):
        
        # safety checks
        assert not any("recon" in p.lower() for p in real_paths), "Real paths contain reconstructions!"
        assert not any("recon" in p.lower() for p in swapped_paths), "fake paths contain reconstructions!"
        
        self.real_paths = real_paths
        self.swapped_paths = swapped_paths
        
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        print(f"[Dataset] {len(real_paths)} real | {len(swapped_paths)} swapped samples loaded.")
    
    def __len__(self):
        return len(self.real_paths) + len(self.swapped_paths)
    
    def __getitem__(self, idx):
        if idx < len(self.real_paths):
            path = self.real_paths[idx]
            label = 0  # real
            
        else:
            path = self.swapped_paths[idx - len(self.real_paths)]
            label = 1  # swapped
            
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        
        return img, label
    
    
# Utils
def get_image_paths(directory, extension=('.jpg', '.png', '.jpeg')):
    paths = []
    for f in os.listdir(directory):
        if f.lower().endswith(extension):
            paths.append(os.path.join(directory, f))
    return sorted(paths)

def deterministic_split(real_paths, swapped_paths, train_ratio, seed_val):
    assert len(real_paths) == len(swapped_paths), "Real and swapped paths must be of equal length."
    
    indices = list(range(len(real_paths)))
    rng = random.Random(seed_val)
    rng.shuffle(indices)
    
    split_idx = int(train_ratio * len(indices))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_real = [real_paths[i] for i in train_idx]
    val_real = [real_paths[i] for i in val_idx]
    train_swapped = [swapped_paths[i] for i in train_idx]
    val_swapped = [swapped_paths[i] for i in val_idx]
    
    return train_real, val_real, train_swapped, val_swapped

# Training / Validation Loop

def train_epoch(
    model,
    loader,
    criterion,
    optimizer, 
    device
):
    
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for imgs, labels in tqdm(loader, desc='training', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return total_loss / len(loader), correct / total


def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='validating', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        
        binary_preds = (torch.tensor(all_probs) >= 0.5).int().numpy()
        acc = accuracy_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        return total_loss / len(loader), acc, f1, auc
    
# Main Training Script
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load image paths
    real_paths = get_image_paths(config['data']['real_dir'])
    swapped_paths = get_image_paths(config['data']['swapped_dir'])
    
    assert len(real_paths) > 0, "No real images found!"
    assert len(swapped_paths) > 0, "No swapped images found!"
    
    # Deterministic split
    train_real, val_real, train_swapped, val_swapped = deterministic_split(
        real_paths,
        swapped_paths,
        train_ratio=0.8,
        seed_val=config['seed']
    )
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = FreqSwapDataset(train_real, train_swapped, transform=transform)
    val_dataset = FreqSwapDataset(val_real, val_swapped, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    
    model = FreqSwapDataset(
        backbone=config['model']['backbone'],
        model_name=config['model'].get('model_name'),
        num_classes=config['model']['num_classes']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr = config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    os.makedirs("runs/latest", exist_ok=True)
    best_auc = 0.0

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_auc = validate(
            model, val_loader, criterion, device
        )

        print(f"\nEpoch [{epoch+1}/{config['training']['epochs']}]")
        print(f"Train  | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val    | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
              f"F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "runs/latest/model.pth")

            with open("runs/latest/metrics.yaml", "w") as f:
                yaml.dump({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_auc": val_auc
                }, f)


if __name__ == "__main__":
    main() 
        

        