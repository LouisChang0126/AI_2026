"""Generate visual examples for the report:
1. Augmentation pairs: original vs two augmented views
2. kNN retrieval: query image + top-k nearest neighbors
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from PIL import Image

from config import Config
from model import get_backbone
from dataset import CIFAR10_MEAN, CIFAR10_STD, SimCLRAugmentation

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "logs", "visuals")
os.makedirs(OUT_DIR, exist_ok=True)

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def denormalize(tensor, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    """Reverse normalization for display."""
    t = tensor.clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    return t.clamp(0, 1)


# =========================================================================
# 1. Augmentation Examples
# =========================================================================
def generate_augmentation_examples():
    """Show original images and their two augmented views."""
    print("Generating augmentation examples...")

    # Load raw CIFAR-10 (no transform)
    raw_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False
    )

    aug = SimCLRAugmentation(image_size=32, dataset="cifar10")

    # Pick one image per class (first occurrence)
    chosen = {}
    for idx in range(len(raw_dataset)):
        img_pil, label = raw_dataset[idx]
        if label not in chosen:
            chosen[label] = (img_pil, idx)
        if len(chosen) == 10:
            break

    # Transposed layout: 3 rows (Original, View1, View2) x 10 columns (classes)
    fig, axes = plt.subplots(3, 10, figsize=(16, 5))
    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    row_labels = ["Original", "View 1", "View 2"]
    to_tensor = T.ToTensor()

    for col, label in enumerate(range(10)):
        img_pil, idx = chosen[label]

        # Original
        orig_tensor = to_tensor(img_pil)
        axes[0, col].imshow(orig_tensor.permute(1, 2, 0).numpy())
        axes[0, col].set_title(f"{CLASSES[label]}", fontsize=8, pad=2)

        # Two augmented views
        view1, view2 = aug(img_pil)
        for r, view in enumerate([view1, view2], 1):
            disp = denormalize(view).permute(1, 2, 0).numpy()
            axes[r, col].imshow(disp)

    # Row labels on left
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=9, rotation=90, labelpad=8)

    # Remove all ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    path = os.path.join(OUT_DIR, "augmentation_examples.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")


# =========================================================================
# 2. kNN Retrieval Examples
# =========================================================================
@torch.no_grad()
def generate_knn_examples():
    """For several query images, show their k nearest neighbors."""
    print("Generating kNN retrieval examples...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()

    # Load backbone
    ckpt_path = os.path.join(BASE, "checkpoints", "baseline_epoch200.pt")
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}, skipping.")
        return

    backbone = get_backbone(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone_state"])
    backbone.eval()

    # Load data with test transform (no augment)
    test_transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=test_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_transform
    )

    # Raw datasets for display
    raw_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=False)
    raw_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=False)

    # Extract train features
    print("  Extracting train features...")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=False, num_workers=0)
    feat_bank, label_bank = [], []
    for imgs, labels in train_loader:
        feats = backbone(imgs.to(device))
        feats = F.normalize(feats, dim=1)
        feat_bank.append(feats.cpu())
        label_bank.append(labels)
    feat_bank = torch.cat(feat_bank, dim=0)
    label_bank = torch.cat(label_bank, dim=0)

    # Extract test features
    print("  Extracting test features...")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    test_feats, test_labels = [], []
    for imgs, labels in test_loader:
        feats = backbone(imgs.to(device))
        feats = F.normalize(feats, dim=1)
        test_feats.append(feats.cpu())
        test_labels.append(labels)
    test_feats = torch.cat(test_feats, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Transposed layout: 6 rows (query + 5 neighbors) x 10 columns (classes)
    k = 5
    query_indices = []
    for c in range(10):
        idx = (test_labels == c).nonzero(as_tuple=True)[0][0].item()
        query_indices.append(idx)

    n_rows = k + 1  # query + k neighbors
    fig, axes = plt.subplots(n_rows, 10, figsize=(16, 10))
    fig.subplots_adjust(wspace=0.05, hspace=0.25)

    for col, q_idx in enumerate(query_indices):
        q_feat = test_feats[q_idx:q_idx+1]
        q_label = test_labels[q_idx].item()

        # Query image (row 0)
        q_img, _ = raw_test[q_idx]
        axes[0, col].imshow(q_img)
        axes[0, col].set_title(f"{CLASSES[q_label]}", fontsize=8, fontweight="bold", pad=2)
        for spine in axes[0, col].spines.values():
            spine.set_edgecolor("blue")
            spine.set_linewidth(2)

        # kNN neighbors (rows 1..k)
        sims = (q_feat @ feat_bank.T).squeeze(0)
        topk_sim, topk_idx = sims.topk(k)

        for r, (sim_val, nb_idx) in enumerate(zip(topk_sim, topk_idx), 1):
            nb_idx = nb_idx.item()
            nb_label = label_bank[nb_idx].item()
            nb_img, _ = raw_train[nb_idx]

            ax = axes[r, col]
            ax.imshow(nb_img)
            match = nb_label == q_label
            color = "green" if match else "red"
            ax.set_title(f"{sim_val:.2f}", fontsize=7, color=color, pad=1)
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)

    # Row labels on left
    axes[0, 0].set_ylabel("Query", fontsize=9, rotation=90, labelpad=8)
    for r in range(1, n_rows):
        axes[r, 0].set_ylabel(f"NN-{r}", fontsize=8, rotation=90, labelpad=8)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    path = os.path.join(OUT_DIR, "knn_retrieval_examples.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")


if __name__ == "__main__":
    generate_augmentation_examples()
    generate_knn_examples()
    print("Done.")
