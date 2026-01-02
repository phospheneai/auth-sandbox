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

# -------------------------
# PATH SETUP
# -------------------------
sys.path.append("../models")
from detector import HFLFDetector

from databunch import DataPipeline, paired_collate_fn


# ============================================================
# RAW VALIDATION DATASET (NO AUGMENTATION, NO MIXING)
# ============================================================
class RawValDataset(Dataset):
    """
    Validation dataset:
      - real images  -> label 0
      - recon images -> label 1
    """

    def __init__(self, real_dir, recon_dir):
        self.samples = []

        real_paths = list(Path(real_dir).glob("*"))
        recon_paths = list(Path(recon_dir).glob("*"))

        for p in real_paths:
            self.samples.append((p, 0))

        for p in recon_paths:
            self.samples.append((p, 1))

        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
        ])

        print("[RawValDataset initialized]")
        print(f"  Real images : {len(real_paths)}")
        print(f"  Fake images : {len(recon_paths)}")
        print(f"  Total       : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# =========================
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
    val_dataset = RawValDataset(
        real_dir=config["data"]["real_dir"],
        recon_dir=config["data"]["recon_dir"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = HFLFDetector(
        backbone=config["model"]["backbone"],
        model_name=config["model"].get("model_name"),
        num_classes=config["model"]["num_classes"]
    ).to(device)

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
    for epoch in range(1, config["training"]["epochs"] + 1):
        print(f"\n>>> Starting Epoch {epoch}/{config['training']['epochs']} - Training...")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"✓ Training complete. Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        print(f">>> Starting Validation...")
        val_loss, val_acc, val_f1, val_auc = validate(
            model, val_loader, criterion, device, epoch
        )
        print(f"✓ Validation complete.")

        print(f"\nEpoch [{epoch}/{config['training']['epochs']}]")
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
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
