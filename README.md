# Waste Classification

ITM-360 Artificial Intelligence project, American University of Phnom Penh.

This project classifies waste photos into four categories: Hazardous, Non-Recyclable, Organic and Recyclable. It measures how well models trained on clean studio images hold up on cluttered real-world images. The result is a decision-support tool: it suggests a category, and the person still makes the final disposal choice.

## Research questions

| | Question | How it's tested |
| --- | --- | --- |
| RQ1 | How much does macro F1 drop from default to real_world images, and which classes are most affected? | `make baseline`, `make transfer` |
| RQ2 | Does transfer learning score higher and have a smaller gap than a basic CNN? | `make compare` |
| RQ3 (optional) | Does fine-tuning with real_world images reduce the gap without hurting default performance? | `make finetune` |

## Project structure

```text
waste-classification/
├── app/
│   └── app.py                     # Upload-a-photo web app
├── configs/
│   ├── class_mapping.yaml         # 30 dataset categories -> 4 classes
│   ├── default.yaml               # Shared settings
│   ├── basic_cnn.yaml             # Model A
│   ├── transfer.yaml              # Model B
│   └── transfer_real_world.yaml   # Optional Step 7
├── data/
│   ├── raw/                       # Kaggle images (not in git)
│   └── splits/manifest.csv        # Class, image type and split for each image (in git)
├── models/                        # Checkpoints (not in git)
├── notebooks/                     # Data exploration and error analysis
├── outputs/runs/                  # Per-epoch logs for each run (not in git)
├── reports/<experiment>/          # Metrics, predictions, confusion matrices
├── src/waste_classifier/
│   ├── compare.py                 # Results table across experiments
│   ├── data.py                    # Dataset, transforms, dataloaders
│   ├── engine.py                  # Shared train/eval loop
│   ├── evaluate.py                # Scores on both test sets
│   ├── model.py                   # Basic CNN, transfer models, checkpoints
│   ├── predict.py                 # Classify one image
│   ├── split.py                   # Duplicate removal and stratified split
│   ├── train.py                   # Training
│   └── utils.py                   # Config, seeding, device selection
├── tests/
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

## Dataset

Download [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) by Alistair King and unzip it into `data/raw/`. It has 15,000 images in 30 categories, and each category has 250 `default` and 250 `real_world` images. `make split` finds the category folders even if they're nested inside the unzipped folder.

The four classes are defined in `configs/class_mapping.yaml`:

| Class | Source categories | Images |
| --- | --- | --- |
| Hazardous | 1 | 500 |
| Non-Recyclable | 8 | 4,000 |
| Organic | 4 | 2,000 |
| Recyclable | 17 | 8,500 |

## How the data is used

`make split` removes byte-identical images, then splits each category and image type 70/15/15:

| Split | default | real_world |
| --- | --- | --- |
| train (70%) | Trains Models A and B | Used only by optional fine-tuning |
| val (15%) | Selects the best epoch | Used only by optional fine-tuning |
| test (15%) | Default test set | Real-world test set |

Models A and B never see a real_world image during training or model selection. Both test sets stay held out in every experiment, so all results can be compared directly.

Commit `data/splits/manifest.csv` after running `make split` once, so the whole team uses the same split. Don't change the test sets after you've started comparing models.

## Usage

```bash
make split      # Build the manifest
make baseline   # Train and evaluate Model A (Basic CNN)
make transfer   # Train and evaluate Model B (Transfer Learning)
make finetune   # Optional: Step 7, requires make transfer first
make compare    # Write reports/comparison.csv
make app        # Launch the upload app

python -m waste_classifier.predict path/to/image.jpg
```

Training uses CUDA or Apple Silicon (MPS) when available, and falls back to the CPU.

Each experiment writes these files to `reports/<experiment>/`:

- `metrics.json`: macro F1 (primary), balanced accuracy, accuracy, per-class scores, the majority-class baseline and the default-to-real_world drop
- `confusion_default.png` and `confusion_real_world.png`
- `predictions.csv`: one row per test image, for error analysis in a notebook

To try another backbone, change `model.name` and `experiment` in a copy of `configs/transfer.yaml`.

## Results

| Model | Default macro F1 | Real-world macro F1 | Drop |
| --- | --- | --- | --- |
| Majority-class baseline | TBD | TBD | TBD |
| Model A: Basic CNN | TBD | TBD | TBD |
| Model B: Transfer learning | TBD | TBD | TBD |

## Limitations

- Hazardous contains only aerosol cans, so the model can't learn other hazardous waste such as batteries.
- The real_world images are curated, so results show robustness to this shift rather than field performance in Phnom Penh.

## Development

```bash
make test
make lint
```
