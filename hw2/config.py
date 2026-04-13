"""Central configuration for all experiments."""

import argparse
from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class Config:
    # --- Data ---
    dataset: str = "cifar10"
    data_dir: str = "./data"
    num_workers: int = 2
    image_size: int = 32

    # --- Model ---
    backbone: str = "resnet18"
    feature_dim: int = 512
    projector_hidden_dim: int = 512
    projector_out_dim: int = 128
    use_projector: bool = True  # set False to ablate projector head

    # --- SimCLR Training ---
    batch_size: int = 256
    ssl_epochs: int = 200
    ssl_lr: float = 3e-4
    ssl_weight_decay: float = 1e-6
    temperature: float = 0.5

    # --- kNN Monitor ---
    knn_k: int = 20
    knn_every: int = 5

    # --- Linear Probing ---
    lp_epochs: int = 100
    lp_lr: float = 1e-3
    lp_weight_decay: float = 1e-6
    lp_dataset: str = "cifar10"  # "cifar10", "cifar100", "stl10"

    # --- Supervised Training ---
    sl_epochs: int = 200
    sl_lr: float = 3e-4
    sl_weight_decay: float = 1e-6
    num_classes: int = 10

    # --- System ---
    device: str = "cuda"
    seed: int = 42
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    resume: Optional[str] = None  # path to checkpoint, or "random"

    # --- Experiment Tag ---
    experiment_name: str = "baseline"


def get_config() -> Config:
    """Parse command-line arguments and return a Config object.

    Every field in Config can be overridden from the CLI, e.g.:
        python train_simclr.py --batch_size 512 --temperature 0.1
    """
    config = Config()
    parser = argparse.ArgumentParser()

    for f in fields(Config):
        ftype = f.type
        # Handle Optional[str]
        if ftype is Optional[str]:
            ftype = str
        if ftype is bool:
            parser.add_argument(
                f"--{f.name}",
                type=lambda v: v.lower() in ("true", "1", "yes"),
                default=getattr(config, f.name),
            )
        else:
            parser.add_argument(
                f"--{f.name}", type=ftype, default=getattr(config, f.name)
            )

    args = parser.parse_args()
    return Config(**vars(args))
