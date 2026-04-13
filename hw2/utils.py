"""Shared utilities: NT-Xent loss, kNN monitor, checkpointing, metrics logging."""

import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ========================================================================
# NT-Xent Loss
# ========================================================================

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    Expects input ``z`` of shape (2N, D) where z[i] and z[i+N] are mates
    (two augmented views of the same source image).  ``z`` should be
    L2-normalised before being passed in.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z: (2N, D) L2-normalised projection vectors.

        Returns:
            Scalar loss averaged over all 2N views.
        """
        device = z.device
        batch_size = z.shape[0] // 2  # N

        # Pairwise cosine similarity (z is unit-norm -> dot product)
        sim = z @ z.T / self.temperature  # (2N, 2N)

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim = sim.masked_fill(mask, -1e9)

        # Positive-pair labels: view i <-> view i+N
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])  # (2N,)

        loss = F.cross_entropy(sim, labels)
        return loss


# ========================================================================
# kNN Monitor
# ========================================================================

@torch.no_grad()
def knn_evaluate(
    backbone: nn.Module,
    memory_loader: DataLoader,
    test_loader: DataLoader,
    k: int = 20,
    num_classes: int = 10,
    device: str = "cuda",
) -> float:
    """k-Nearest-Neighbour classification accuracy on the test set.

    1. Extract L2-normalised features for the full training set (memory bank).
    2. For each test image, find k nearest neighbours and do weighted voting.
    3. Return top-1 accuracy.
    """
    backbone.eval()

    # Build memory bank from training set
    feature_bank, label_bank = [], []
    for images, labels in memory_loader:
        features = backbone(images.to(device))
        features = F.normalize(features, dim=1)
        feature_bank.append(features)
        label_bank.append(labels.to(device))

    feature_bank = torch.cat(feature_bank, dim=0)  # (M, 512)
    label_bank = torch.cat(label_bank, dim=0)       # (M,)

    # Evaluate on test set
    correct, total = 0, 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        features = backbone(images)
        features = F.normalize(features, dim=1)

        # Cosine similarity with memory bank
        sim = features @ feature_bank.T  # (B, M)
        topk_sim, topk_idx = sim.topk(k, dim=1)  # (B, k)
        topk_labels = label_bank[topk_idx]  # (B, k)

        # Weighted voting: accumulate similarity for each class
        votes = torch.zeros(images.size(0), num_classes, device=device)
        votes.scatter_add_(1, topk_labels, topk_sim)

        preds = votes.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ========================================================================
# Seed
# ========================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================================================================
# Checkpointing
# ========================================================================

def save_checkpoint(state: dict, filepath: str):
    """Save a checkpoint dictionary to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"  -> Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, device: str = "cuda") -> dict:
    """Load a checkpoint dictionary from disk."""
    state = torch.load(filepath, map_location=device, weights_only=False)
    print(f"  -> Checkpoint loaded: {filepath}")
    return state


# ========================================================================
# Metrics Logger
# ========================================================================

class MetricsLogger:
    """Accumulate per-epoch metrics and produce plots / CSV."""

    def __init__(self):
        self.history: dict[str, list] = defaultdict(list)
        self.epochs: dict[str, list] = defaultdict(list)

    def log(self, epoch: int, **kwargs):
        """Log one or more named metrics for the given epoch."""
        for name, value in kwargs.items():
            self.history[name].append(value)
            self.epochs[name].append(epoch)

    def save_plot(self, metrics: list[str], filepath: str, title: str = "",
                  ylabel: str = ""):
        """Plot specified metrics vs. epoch and save to file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for m in metrics:
            ax.plot(self.epochs[m], self.history[m], label=m)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel or ", ".join(metrics))
        ax.set_title(title or ", ".join(metrics))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        print(f"  -> Plot saved: {filepath}")

    def save_csv(self, filepath: str):
        """Export all metrics to CSV."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        all_metrics = sorted(self.history.keys())
        # Find the union of all epochs
        all_epochs = sorted(set(e for ep_list in self.epochs.values()
                                for e in ep_list))
        # Build lookup: metric -> {epoch: value}
        lookup = {}
        for m in all_metrics:
            lookup[m] = dict(zip(self.epochs[m], self.history[m]))

        with open(filepath, "w") as f:
            f.write("epoch," + ",".join(all_metrics) + "\n")
            for ep in all_epochs:
                vals = [str(lookup[m].get(ep, "")) for m in all_metrics]
                f.write(f"{ep}," + ",".join(vals) + "\n")
        print(f"  -> CSV saved: {filepath}")
