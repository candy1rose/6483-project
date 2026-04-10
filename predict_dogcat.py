from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.data import create_dataloaders, parse_prediction_ids
from src.models import build_model
from src.utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Dogs vs. Cats submission.csv predictions.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=str, default="outputs/submissions/submission.csv")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = checkpoint["image_size"]
    model_name = checkpoint["model_name"]
    pretrained = checkpoint.get("pretrained", False)

    bundle = create_dataloaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.2,
        use_augmentation=False,
        train_limit=args.train_samples,
        val_limit=args.val_samples,
        seed=checkpoint.get("args", {}).get("seed", 42),
    )
    if bundle.test_loader is None:
        raise FileNotFoundError("Test directory was not found. Expected data/test with raw test images.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = build_model(model_name, num_classes=2, pretrained=pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_ids: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for images, stems in tqdm(bundle.test_loader):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_ids.extend(parse_prediction_ids(stems))

    df = pd.DataFrame({"id": all_ids, "label": all_preds}).sort_values("id")
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    main()
