from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_loaders(
    data_dir: str | Path,
    batch_size: int = 64,
    val_fraction: float = 0.10,
    train_subset: int | None = None,
    seed: int = 42,
    num_workers: int = 0,
):
    """Load Fashion-MNIST and return train, validation, and test loaders."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    data_dir = Path(data_dir)
    train_full = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(seed)
    if train_subset is not None and train_subset < len(train_full):
        indices = torch.randperm(len(train_full), generator=generator)[:train_subset]
        train_full = Subset(train_full, indices.tolist())

    val_size = int(len(train_full) * val_fraction)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(
        train_full, [train_size, val_size], generator=generator
    )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader
