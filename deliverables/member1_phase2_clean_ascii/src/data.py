from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader | None
    train_size: int
    val_size: int
    test_size: int
    class_names: list[str]


class TestImageDataset(Dataset):
    def __init__(self, root: str | Path, transform: transforms.Compose):
        self.root = Path(root)
        self.transform = transform
        self.samples = sorted(
            path for path in self.root.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not self.samples:
            raise FileNotFoundError(f"No test images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.stem


def build_train_transform(image_size: int, use_augmentation: bool) -> transforms.Compose:
    ops: list[transforms.Compose | transforms.Resize | transforms.Normalize] = []
    if use_augmentation:
        ops.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            ]
        )
    else:
        ops.extend([transforms.Resize((image_size, image_size))])

    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _resolve_split_dirs(data_dir: Path) -> tuple[Path, Path | None, Path | None]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Expected training directory at {train_dir}. "
            "Dataset layout should look like data/train/{cat,dog}, optional data/val/{cat,dog}, and data/test/."
        )
    return train_dir, val_dir if val_dir.exists() else None, test_dir if test_dir.exists() else None


def create_dataloaders(
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    use_augmentation: bool,
    train_limit: int | None = None,
    val_limit: int | None = None,
    seed: int = 42,
) -> DatasetBundle:
    data_dir = Path(data_dir)
    train_dir, val_dir, test_dir = _resolve_split_dirs(data_dir)

    train_transform = build_train_transform(image_size, use_augmentation)
    eval_transform = build_eval_transform(image_size)

    if val_dir is not None:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    else:
        base_dataset = datasets.ImageFolder(train_dir)
        val_size = max(1, int(len(base_dataset) * val_split))
        train_size = len(base_dataset) - val_size
        if train_size <= 0:
            raise ValueError("Validation split is too large for the available training data.")
        indices = torch.randperm(len(base_dataset)).tolist()
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = Subset(datasets.ImageFolder(train_dir, transform=train_transform), train_indices)
        val_dataset = Subset(datasets.ImageFolder(train_dir, transform=eval_transform), val_indices)

    if train_limit is not None:
        train_dataset = _limit_dataset(train_dataset, train_limit, seed)
    if val_limit is not None:
        val_dataset = _limit_dataset(val_dataset, val_limit, seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    test_size = 0
    if test_dir is not None:
        test_dataset = TestImageDataset(test_dir, transform=eval_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_size = len(test_dataset)

    class_names = list(getattr(getattr(train_dataset, "dataset", train_dataset), "classes", []))
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=test_size,
        class_names=class_names,
    )


def parse_prediction_ids(stems: Iterable[str]) -> list[int]:
    ids: list[int] = []
    for stem in stems:
        digits = "".join(ch for ch in stem if ch.isdigit())
        ids.append(int(digits) if digits else int(stem))
    return ids


def _get_targets(dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_targets = _get_targets(dataset.dataset)
        return [base_targets[idx] for idx in dataset.indices]
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "samples"):
        return [label for _, label in dataset.samples]
    raise TypeError("Dataset does not expose class targets.")


def _limit_dataset(dataset, limit: int, seed: int):
    if limit <= 0:
        raise ValueError("Sample limit must be positive.")
    dataset_size = len(dataset)
    if limit > dataset_size:
        raise ValueError(f"Requested {limit} samples from a dataset of size {dataset_size}.")
    if limit == dataset_size:
        return dataset

    targets = _get_targets(dataset)
    generator = torch.Generator().manual_seed(seed)
    per_class_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(targets):
        per_class_indices.setdefault(label, []).append(idx)

    num_classes = len(per_class_indices)
    base_quota = limit // num_classes
    remainder = limit % num_classes

    selected: list[int] = []
    for rank, label in enumerate(sorted(per_class_indices)):
        indices = per_class_indices[label]
        shuffled = torch.tensor(indices)[torch.randperm(len(indices), generator=generator)].tolist()
        quota = base_quota + (1 if rank < remainder else 0)
        if quota > len(shuffled):
            raise ValueError(f"Class {label} does not have enough samples for the requested limit.")
        selected.extend(shuffled[:quota])

    selected = torch.tensor(selected)[torch.randperm(len(selected), generator=generator)].tolist()
    return Subset(dataset, selected)
