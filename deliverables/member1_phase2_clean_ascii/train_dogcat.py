from __future__ import annotations

import argparse
from pathlib import Path
from time import time

import torch
from torch import nn
from torch.optim import Adam, SGD
from tqdm import tqdm

from src.data import create_dataloaders
from src.models import build_model
from src.utils import ensure_dir, plot_training_history, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dogs vs. Cats models for EE6483 Project 2.")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset root containing train/ val/ test/ folders.")
    parser.add_argument("--model", type=str, default="resnet18", choices=["simple_cnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=2000, help="Number of training images to use.")
    parser.add_argument("--val-samples", type=int, default=1000, help="Number of validation images to use.")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ImageNet weights for ResNet18.")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures")
    parser.add_argument("--experiment-name", type=str, default=None)
    return parser.parse_args()


def build_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    progress = tqdm(loader, leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        total_seen += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        progress.set_postfix(loss=f"{total_loss / total_seen:.4f}", acc=f"{total_correct / total_seen:.4f}")

    return total_loss / total_seen, total_correct / total_seen


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    run_name = args.experiment_name or f"{args.model}_img{args.image_size}_bs{args.batch_size}_lr{args.lr}"
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    figure_dir = ensure_dir(args.figure_dir)

    data = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        use_augmentation=not args.no_augmentation,
        train_limit=args.train_samples,
        val_limit=args.val_samples,
        seed=args.seed,
    )

    model = build_model(args.model, num_classes=2, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = checkpoint_dir / f"{run_name}_best.pt"
    start_time = time()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_epoch(model, data.train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, data.val_loader, criterion, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "class_names": data.class_names,
                    "image_size": args.image_size,
                    "pretrained": args.pretrained,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                best_path,
            )

    metrics = {
        "run_name": run_name,
        "train_size": data.train_size,
        "val_size": data.val_size,
        "test_size": data.test_size,
        "class_names": data.class_names,
        "best_val_acc": best_val_acc,
        "history": history,
        "device": str(device),
        "elapsed_seconds": round(time() - start_time, 2),
    }
    save_json(metrics, checkpoint_dir / f"{run_name}_metrics.json")
    plot_training_history(history, figure_dir / f"{run_name}_curves.png")

    print(f"Best checkpoint saved to: {best_path}")
    print(f"Training curves saved to: {figure_dir / f'{run_name}_curves.png'}")


if __name__ == "__main__":
    main()
