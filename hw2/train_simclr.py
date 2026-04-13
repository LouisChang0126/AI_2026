"""SimCLR self-supervised training with kNN monitoring.

Usage:
    python train_simclr.py                              # baseline
    python train_simclr.py --temperature 0.1 --experiment_name temp01
    python train_simclr.py --batch_size 512 --experiment_name bs512
    python train_simclr.py --use_projector False --experiment_name no_proj
"""

import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import get_config
from dataset import get_simclr_loaders, _get_num_classes
from model import SimCLRModel
from utils import (
    NTXentLoss,
    MetricsLogger,
    knn_evaluate,
    save_checkpoint,
    set_seed,
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one epoch of SimCLR training. Returns average loss."""
    model.train()
    total_loss = 0.0

    for (x1, x2), _ in tqdm(loader, desc="  train", leave=False):
        # Concatenate two views: (N,C,H,W) x 2 -> (2N,C,H,W)
        x = torch.cat([x1, x2], dim=0).to(device)

        # Forward pass
        _features, projections = model(x)  # (2N, 512), (2N, 128)
        z = F.normalize(projections, dim=1)

        loss = criterion(z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    config = get_config()
    set_seed(config.seed)
    print(f"=== SimCLR Training: {config.experiment_name} ===")
    print(f"    batch_size={config.batch_size}  temperature={config.temperature}  "
          f"epochs={config.ssl_epochs}  use_projector={config.use_projector}")

    # Data
    ssl_loader, memory_loader, test_loader = get_simclr_loaders(config)
    num_classes = _get_num_classes(config.dataset)

    # Model
    model = SimCLRModel(config).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.ssl_lr, weight_decay=config.ssl_weight_decay
    )
    criterion = NTXentLoss(temperature=config.temperature)
    logger = MetricsLogger()

    start_epoch = 1
    # Resume from checkpoint if specified
    if config.resume and config.resume != "random":
        ckpt = torch.load(config.resume, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from epoch {ckpt['epoch']}")

    # Training loop
    for epoch in range(start_epoch, config.ssl_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, ssl_loader, optimizer, criterion, config.device
        )
        elapsed = time.time() - t0
        logger.log(epoch, ssl_loss=train_loss)

        # kNN monitor
        knn_str = ""
        if epoch % config.knn_every == 0 or epoch == 1:
            knn_acc = knn_evaluate(
                model.backbone, memory_loader, test_loader,
                k=config.knn_k, num_classes=num_classes, device=config.device,
            )
            logger.log(epoch, knn_accuracy=knn_acc)
            knn_str = f"  kNN={knn_acc:.2%}"

        print(f"Epoch [{epoch}/{config.ssl_epochs}]  "
              f"Loss={train_loss:.4f}{knn_str}  ({elapsed:.1f}s)")

        # Checkpoint
        if epoch % 50 == 0 or epoch == config.ssl_epochs:
            ckpt_path = os.path.join(
                config.checkpoint_dir,
                f"{config.experiment_name}_epoch{epoch}.pt",
            )
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "backbone_state": model.backbone.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": vars(config),
            }, ckpt_path)

    # Save plots and CSV
    log_dir = os.path.join(config.log_dir, config.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    logger.save_plot(
        ["ssl_loss"],
        os.path.join(log_dir, "loss_curve.png"),
        title=f"SimCLR Loss ({config.experiment_name})",
        ylabel="NT-Xent Loss",
    )
    logger.save_plot(
        ["knn_accuracy"],
        os.path.join(log_dir, "knn_curve.png"),
        title=f"kNN Accuracy ({config.experiment_name})",
        ylabel="Top-1 Accuracy",
    )
    logger.save_csv(os.path.join(log_dir, "metrics.csv"))

    print("Done.")


if __name__ == "__main__":
    main()
