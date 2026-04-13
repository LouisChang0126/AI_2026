"""Linear probing evaluation.

Freezes a pretrained backbone and trains a single linear layer for classification.

Usage:
    # Evaluate SSL model
    python linear_eval.py --resume checkpoints/baseline_epoch200.pt --experiment_name ssl_probe

    # Evaluate supervised model
    python linear_eval.py --resume checkpoints/supervised_epoch200.pt --experiment_name sl_probe

    # Random baseline (no training, just random weights)
    python linear_eval.py --resume random --experiment_name random_probe

    # Transfer to CIFAR-100
    python linear_eval.py --resume checkpoints/baseline_epoch200.pt \
        --lp_dataset cifar100 --experiment_name ssl_cifar100
"""

import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_config
from dataset import get_linear_eval_loaders, _get_num_classes
from model import LinearClassifier, get_backbone
from utils import MetricsLogger, save_checkpoint, set_seed, load_checkpoint


@torch.no_grad()
def evaluate(backbone, classifier, loader, device):
    """Evaluate accuracy with frozen backbone + linear classifier."""
    backbone.eval()
    classifier.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        features = backbone(images)
        logits = classifier(features)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main():
    config = get_config()
    set_seed(config.seed)

    lp_dataset = config.lp_dataset
    num_classes = _get_num_classes(lp_dataset)

    print(f"=== Linear Probing: {config.experiment_name} ===")
    print(f"    backbone from: {config.resume}")
    print(f"    eval dataset: {lp_dataset}  num_classes={num_classes}")
    print(f"    epochs={config.lp_epochs}  lr={config.lp_lr}")

    # Load backbone
    backbone = get_backbone(config).to(config.device)

    if config.resume is None or config.resume == "random":
        print("  Using randomly initialised backbone (lower-bound baseline).")
    else:
        ckpt = load_checkpoint(config.resume, config.device)
        backbone.load_state_dict(ckpt["backbone_state"])
        print("  Backbone weights loaded successfully.")

    # Freeze backbone
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # Data
    train_loader, test_loader = get_linear_eval_loaders(config)

    # Linear classifier
    classifier = LinearClassifier(config.feature_dim, num_classes).to(config.device)
    optimizer = torch.optim.Adam(
        classifier.parameters(), lr=config.lp_lr, weight_decay=config.lp_weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    logger = MetricsLogger()

    # Training loop
    best_acc = 0.0
    for epoch in range(1, config.lp_epochs + 1):
        t0 = time.time()

        # Train linear layer
        classifier.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc="  train", leave=False):
            images, labels = images.to(config.device), labels.to(config.device)
            with torch.no_grad():
                features = backbone(images)
            logits = classifier(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate(backbone, classifier, test_loader, config.device)
        elapsed = time.time() - t0
        best_acc = max(best_acc, test_acc)

        logger.log(epoch, lp_loss=avg_loss, lp_test_acc=test_acc)

        if epoch % 10 == 0 or epoch == config.lp_epochs:
            print(f"Epoch [{epoch}/{config.lp_epochs}]  "
                  f"Loss={avg_loss:.4f}  TestAcc={test_acc:.2%}  ({elapsed:.1f}s)")

    # Save plots and CSV
    log_dir = os.path.join(config.log_dir, config.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    logger.save_plot(
        ["lp_test_acc"],
        os.path.join(log_dir, "linear_probe_acc.png"),
        title=f"Linear Probing ({config.experiment_name})",
        ylabel="Test Accuracy",
    )
    logger.save_csv(os.path.join(log_dir, "metrics.csv"))

    # Save final classifier
    ckpt_path = os.path.join(
        config.checkpoint_dir, f"{config.experiment_name}_linear.pt"
    )
    save_checkpoint({
        "classifier_state": classifier.state_dict(),
        "best_acc": best_acc,
        "config": vars(config),
    }, ckpt_path)

    print(f"\nFinal test accuracy: {test_acc:.2%}  (best: {best_acc:.2%})")
    print("Done.")


if __name__ == "__main__":
    main()
