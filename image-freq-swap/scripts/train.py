import warnings
warnings.filterwarnings(
    "ignore",
    message="Failed to load image Python extension"
)

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disables the warning (use with caution!)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# -------------------------
# PATH SETUP
# -------------------------
# Replace lines 25-27 with:
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from detector import HFLFDetector

from databunch import DataPipeline, paired_collate_fn


# ============================================================
# RAW VALIDATION DATASET (NO AUGMENTATION, NO MIXING)
# ============================================================
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

class SimpleValDataset(Dataset):
    """
    Dataset:
      - images with 'fake' in path -> label 1
      - all other images           -> label 0
      - pad if smaller than target size, center crop if larger
      - recursive search through nested folders
    """

    def __init__(self, root_dir=r"E:/data/Image_freq_val"):
        self.samples = []
        self.size = 336

        # Recursively collect all files in nested folders
        all_paths = list(Path(root_dir).rglob("*"))

        # Filter only image files
        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        all_paths = [p for p in all_paths if p.suffix.lower() in valid_exts]

        for p in all_paths:
            label = 1 if "fake" in str(p).lower() else 0
            self.samples.append((p, label))

        print("[SimpleValDataset initialized]")
        print(f"  Total images : {len(self.samples)}")
        print(f"  Fake images  : {sum(l for _, l in self.samples)}")
        print(f"  Real images  : {len(self.samples) - sum(l for _, l in self.samples)}")

        # Define transforms including DINOv2 normalization
        self.transform = T.Compose([
            T.CenterCrop(self.size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Skipping unreadable file: {path} ({e})")
            # Return a black image of the right size instead of crashing
            img = Image.new("RGB", (self.size, self.size), (0, 0, 0))

        w, h = img.size
        pad_w = max(0, self.size - w)
        pad_h = max(0, self.size - h)

        # Pad if smaller
        if pad_w > 0 or pad_h > 0:
            padding = (pad_w // 2, pad_h // 2,
                       pad_w - pad_w // 2, pad_h - pad_h // 2)
            img = T.Pad(padding, fill=0)(img)

        # Apply transform (crop + tensor + normalize)
        img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)
# TRAIN FUNCTION
# =========================
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for imgs, labels in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader), correct / total


# =========================
# VALIDATION FUNCTION
# =========================
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)

    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    binary_preds = (torch.tensor(all_probs) >= 0.5).int().numpy()

    acc = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return total_loss / len(loader), acc, f1, auc


# =========================
# MAIN
# =========================
def main():
    # -------------------------
    # CONFIG
    # -------------------------
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # TRAIN DATASET (AUGMENTED)
    # -------------------------
    train_dataset = DataPipeline(
        real_dir=config["data"]["real_dir"],
        recon_dir=config["data"]["recon_dir"],
        freq_ratios=tuple(config["data"].get("freq_ratios", (0.0, 0.85))),
        pixel_ratios=tuple(config["data"].get("pixel_ratios", (0.5, 1.0))),
        seed=config["seed"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=paired_collate_fn
    )

    # -------------------------
    # VALIDATION DATASET (RAW)
    # -------------------------
    val_dataset = SimpleValDataset()
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # -------------------------
    # MODEL
    # -------------------------
    # model = HFLFDetector(
    #     backbone=config["model"]["backbone"],
    #     model_name=config["model"].get("model_name"),
    #     num_classes=config["model"]["num_classes"]
    # ).to(device)
    from detectorlora import DINOv2ModelWithLoRA,LoRALinear
    model = DINOv2ModelWithLoRA().to(device)

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            print("LoRA applied to:", name)


    # model.load_state_dict(torch.load("../runs/latest/model.pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    os.makedirs("../runs/latest", exist_ok=True)
    best_auc = 0.0

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    # -------------------------
# TRAINING LOOP
# -------------------------
    print(len(val_dataset))
    for epoch in range(1, config["training"]["epochs"] + 1):


        print(f"\n>>> Starting Epoch {epoch}/{config['training']['epochs']} - Training...")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        torch.save(model.state_dict(), "../runs/latest/checkpoint.pth")

        print(f"✓ Training complete. Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        print(f"\nEpoch [{epoch}/{config['training']['epochs']}]")
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        print(f">>> Starting Validation...")
        val_loss, val_acc, val_f1, val_auc = validate(
            model, val_loader, criterion, device, epoch
        )
        print(f"✓ Validation complete.")
        print(
            f"Val   | Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "../runs/latest/model.pth")

            with open("../runs/latest/metrics.yaml", "w") as f:
                yaml.dump({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "val_auc": val_auc
                }, f)

    print("\n✅ Training completed")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()
