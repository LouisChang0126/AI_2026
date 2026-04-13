"""Data loading and augmentation pipelines for SimCLR, supervised, and linear probing."""

from typing import Tuple

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import Config

# ---------- Normalization stats ----------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

STL10_MEAN = (0.4408, 0.4279, 0.3867)
STL10_STD = (0.2682, 0.2610, 0.2686)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_normalize(dataset: str):
    stats = {
        "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
        "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
        "stl10": (STL10_MEAN, STL10_STD),
    }
    mean, std = stats.get(dataset, (IMAGENET_MEAN, IMAGENET_STD))
    return T.Normalize(mean=mean, std=std)


# ---------- Augmentations ----------

class SimCLRAugmentation:
    """Returns two independently augmented views of the same image."""

    def __init__(self, image_size: int = 32, dataset: str = "cifar10"):
        self.transform = T.Compose([
            T.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4,
                              saturation=0.4, hue=0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            _get_normalize(dataset),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def _get_test_transform(image_size: int = 32, dataset: str = "cifar10"):
    """Deterministic transform for evaluation (no augmentation)."""
    ops = []
    # Resize for datasets with images larger than 32x32
    if dataset in ("stl10",):
        ops.append(T.Resize(image_size))
    ops.append(T.ToTensor())
    ops.append(_get_normalize(dataset))
    return T.Compose(ops)


def _get_supervised_train_transform(image_size: int = 32, dataset: str = "cifar10"):
    """Standard supervised training augmentation."""
    return T.Compose([
        T.RandomCrop(image_size, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        _get_normalize(dataset),
    ])


# ---------- Dataset factories ----------

def _get_num_classes(dataset: str) -> int:
    return {"cifar10": 10, "cifar100": 100, "stl10": 10}.get(dataset, 10)


def _load_dataset(name: str, root: str, train: bool, transform):
    """Load a torchvision dataset by name."""
    if name == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=root, train=train, transform=transform, download=True
        )
    elif name == "cifar100":
        return torchvision.datasets.CIFAR100(
            root=root, train=train, transform=transform, download=True
        )
    elif name == "stl10":
        split = "train" if train else "test"
        return torchvision.datasets.STL10(
            root=root, split=split, transform=transform, download=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")


# ---------- DataLoader factories ----------

def get_simclr_loaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (ssl_train_loader, memory_loader, test_loader).

    - ssl_train_loader: training set with SimCLR dual augmentation.
    - memory_loader:    training set with test transform (for kNN memory bank).
    - test_loader:      test set with test transform.
    """
    ssl_transform = SimCLRAugmentation(config.image_size, config.dataset)
    test_transform = _get_test_transform(config.image_size, config.dataset)

    train_ssl = _load_dataset(config.dataset, config.data_dir, train=True,
                              transform=ssl_transform)
    train_mem = _load_dataset(config.dataset, config.data_dir, train=True,
                              transform=test_transform)
    test_set = _load_dataset(config.dataset, config.data_dir, train=False,
                             transform=test_transform)

    ssl_train_loader = DataLoader(
        train_ssl, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    memory_loader = DataLoader(
        train_mem, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    return ssl_train_loader, memory_loader, test_loader


def get_supervised_loaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for supervised training."""
    train_transform = _get_supervised_train_transform(
        config.image_size, config.dataset
    )
    test_transform = _get_test_transform(config.image_size, config.dataset)

    train_set = _load_dataset(config.dataset, config.data_dir, train=True,
                              transform=train_transform)
    test_set = _load_dataset(config.dataset, config.data_dir, train=False,
                             transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def get_linear_eval_loaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for linear probing.

    Uses ``config.lp_dataset`` which may differ from the SSL training dataset.
    """
    dataset_name = config.lp_dataset
    test_transform = _get_test_transform(config.image_size, dataset_name)

    # Linear probing uses deterministic transform (no augmentation) for both
    train_set = _load_dataset(dataset_name, config.data_dir, train=True,
                              transform=test_transform)
    test_set = _load_dataset(dataset_name, config.data_dir, train=False,
                             transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    return train_loader, test_loader
