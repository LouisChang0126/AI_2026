"""Supervised baseline training from scratch.

Usage:
    python train_supervised.py --batch_size 256 --experiment_name supervised
"""

import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_config
from dataset import get_supervised_loaders, _get_num_classes
from model import SupervisedModel
from utils import MetricsLogger, save_checkpoint, set_seed


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One epoch of supervised training. Returns (avg_loss, train_accuracy)."""
    model.train()
    total_loss = 0.0
    correct, total = 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate classification accuracy on a dataset."""
    model.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main():
    config = get_config()
    set_seed(config.seed)
    config.num_classes = _get_num_classes(config.dataset)
    print(f"=== Supervised Training: {config.experiment_name} ===")
    print(f"    batch_size={config.batch_size}  epochs={config.sl_epochs}  "
          f"num_classes={config.num_classes}")

    # Data
    train_loader, test_loader = get_supervised_loaders(config)

    # Model
    model = SupervisedModel(config).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.sl_lr, weight_decay=config.sl_weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger()

    # Training loop
    for epoch in range(1, config.sl_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, config.device
        )
        test_acc = evaluate(model, test_loader, config.device)
        elapsed = time.time() - t0

        logger.log(epoch, sl_loss=train_loss, sl_train_acc=train_acc,
                    sl_test_acc=test_acc)

        print(f"Epoch [{epoch}/{config.sl_epochs}]  "
              f"Loss={train_loss:.4f}  TrainAcc={train_acc:.2%}  "
              f"TestAcc={test_acc:.2%}  ({elapsed:.1f}s)")

        # Checkpoint
        if epoch % 50 == 0 or epoch == config.sl_epochs:
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
        ["sl_loss"],
        os.path.join(log_dir, "loss_curve.png"),
        title=f"Supervised Loss ({config.experiment_name})",
        ylabel="Cross-Entropy Loss",
    )
    logger.save_plot(
        ["sl_train_acc", "sl_test_acc"],
        os.path.join(log_dir, "accuracy_curve.png"),
        title=f"Supervised Accuracy ({config.experiment_name})",
        ylabel="Accuracy",
    )
    logger.save_csv(os.path.join(log_dir, "metrics.csv"))

    print(f"Final test accuracy: {test_acc:.2%}")
    print("Done.")


if __name__ == "__main__":
    main()
