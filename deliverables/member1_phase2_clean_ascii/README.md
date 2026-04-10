# EE6483 Project 2

This workspace now contains the Phase 2 `Dogs vs. Cats` main-task code for the team member who is responsible for model building and training.

The current default setup uses exactly `2000` training images and `1000` validation images, matching the requested experiment size.

## What is included

- `train_dogcat.py`: trains either a custom CNN baseline or a ResNet18 model
- `predict_dogcat.py`: loads the best checkpoint and exports `submission.csv`
- `src/data.py`: data loading, preprocessing, augmentation, and test-set parsing
- `src/models.py`: `SimpleCNN` and transfer-learning `ResNet18`
- `src/utils.py`: random seed control, output helpers, and training-curve plotting
- `outputs/checkpoints/`: best model checkpoints and metrics JSON
- `outputs/figures/`: loss/accuracy curves for the report
- `outputs/submissions/`: Kaggle-format prediction file

## Recommended dataset layout

```text
data/
├── train/
│   ├── cat/
│   └── dog/
├── val/
│   ├── cat/
│   └── dog/
└── test/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

If `data/val/` is missing, the training script will split the training set automatically according to `--val-split`.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training examples

Custom CNN baseline:

```bash
python3 train_dogcat.py \
  --data-dir datasets \
  --model simple_cnn \
  --epochs 12 \
  --batch-size 32 \
  --lr 1e-3 \
  --image-size 128
```

Transfer learning with ResNet18:

```bash
python3 train_dogcat.py \
  --data-dir datasets \
  --model resnet18 \
  --pretrained \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --image-size 224
```

Useful knobs for the comparison experiments in the report:

- `--optimizer adam` or `--optimizer sgd`
- `--batch-size 32` or `64`
- `--lr 1e-3` or `1e-4`
- `--image-size 128` or `224`
- add `--no-augmentation` to measure augmentation effects

## Generate submission.csv

```bash
python3 predict_dogcat.py \
  --data-dir datasets \
  --checkpoint outputs/checkpoints/resnet18_img224_bs32_lr0.0001_best.pt \
  --output outputs/submissions/submission.csv
```

The output format matches `sampleSubmission.csv`:

```csv
id,label
1,0
2,1
```

where `0 = cat` and `1 = dog`.

## What to include in the report for Member 1

- Baseline CNN architecture and training settings
- ResNet18 fine-tuning setup
- Hyperparameter comparison table
- Best validation accuracy
- Training/validation curves from `outputs/figures/`
- Best checkpoint path and the generated `submission.csv`

## Current assumptions

- The dataset is now expected under `datasets/`.
- The local environment currently does not have PyTorch installed yet.
- Until `torch` and `torchvision` are installed, the code cannot be trained in this workspace.
