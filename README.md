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
│   ├── app.py                     # Upload-a-photo web app
│   └── live_camera.py             # Live camera window with object tracking
├── configs/
│   ├── class_mapping.yaml         # 30 dataset categories -> 4 classes
│   ├── default.yaml               # Shared settings
│   ├── basic_cnn.yaml             # Model A
│   ├── basic_cnn_strong_weights.yaml  # First Model A run
│   ├── transfer.yaml              # Model B
│   ├── transfer_real_world.yaml   # Optional Step 7
│   └── transfer_large.yaml        # Final model for the app
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
│   ├── near_duplicates.py         # Flags test images that nearly match training images
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
| test (15%) | Default test set | Held-out real-world test set |

Models A and B never see a real_world image during training or model selection, so they're also scored on all unique real_world images. That larger set is used for RQ1 and RQ2, because the held-out set has only 17 Hazardous images. RQ3 uses the held-out set, because fine-tuning trains on the rest.

The dataset has 15,000 files, but only 11,263 are unique: 7,020 default and 4,243 real_world. Many images appear in both folders, and removing them stops the same image being used for both training and testing. `make split` also flags near-copies, such as resized versions of the same photo, by comparing 32x32 thumbnails. Near-copies of images a model could have trained on are left out of every test set. The cutoff (`data.near_duplicate_rmse`) was chosen by inspecting image pairs, and some near-copies above it may remain.

Commit `data/splits/manifest.csv` after running `make split` once, so the whole team uses the same split. Don't change the test sets after you've started comparing models.

## Usage

```bash
make split      # Build the manifest
make baseline   # Train and evaluate Model A (Basic CNN)
make transfer   # Train and evaluate Model B (Transfer Learning)
make finetune   # Optional: Step 7, requires make transfer first
make final      # Train and evaluate the final model
make compare    # Write reports/comparison.csv
make app        # Launch the upload app (requires make final first)
make camera     # Open the live camera window

python -m waste_classifier.predict path/to/image.jpg
```

The live camera window follows the item you're showing and ignores the rest of the room:

1. It learns the room for about a second when it starts, so keep items out of view until then.
2. Only something new, or something a moving hand is holding, can be tracked. Your body is masked out.
3. Once it locks on, the box stays with that item until it leaves.

A new result appears about every 1.5 seconds, averaged over several frames, and the waste model only sees what's inside the box. MediaPipe's models download once, about 37 MB, on first run.

Click any object to follow it, even one that was already in the room. Press `R` to reset and relearn the room, `1` to `5` to switch models, and `Q` or `Esc` to quit. It says "Not sure" below 60% confidence, a starting value you can change with `--min-confidence`. It's a demo only, so its results aren't part of the reported scores.

Training uses CUDA or Apple Silicon (MPS) when available, and falls back to the CPU.

Each experiment writes these files to `reports/<experiment>/`:

- `metrics.json`: macro F1 (primary), balanced accuracy, accuracy, per-class scores, the majority-class baseline and the default-to-real_world drop
- `confusion_<test set>.png`: one confusion matrix per test set
- `predictions.csv`: one row per test image, for error analysis in a notebook

To try another backbone, change `model.name` and `experiment` in a copy of `configs/transfer.yaml`.

Class weighting is controlled by `train.class_weight_power`. Full weighting (`1.0`) made the first Model A run over-predict Hazardous, so the default is `0.5`. Choose this setting using validation scores, not test scores, and use the same value for every model you compare.

## Results

All scores are from one training run per model, with near-copies of training images removed from the test sets.

**RQ1 and RQ2: models trained on default images only**, scored on 790 default test images and 3,843 real_world images:

| Model | Default macro F1 | Real-world macro F1 | Drop | Default accuracy | Real-world accuracy |
| --- | --- | --- | --- | --- | --- |
| Majority-class baseline | 0.189 | 0.192 | | 60.8% | 62.4% |
| Model A: Basic CNN | 0.592 | 0.483 | 0.109 | 71.1% | 64.8% |
| Model B: EfficientNet-B0 | 0.916 | 0.775 | 0.140 | 94.1% | 86.7% |

**RQ3 and final model**, scored on 790 default test images and 503 held-out real_world images:

| Model | Trained on | Default macro F1 | Real-world macro F1 | Drop | Default accuracy | Real-world accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| Model B: EfficientNet-B0 | default | 0.916 | 0.789 | 0.126 | 94.1% | 87.3% |
| Model B fine-tuned | default, then both | 0.933 | 0.851 | 0.082 | 94.8% | 91.3% |
| Final: EfficientNet-B2 | both | 0.938 | 0.887 | 0.051 | 95.4% | 91.9% |

The final model differs from Model B in both size and training data, so use the fine-tuned Model B to answer RQ3.

## Limitations

- Hazardous contains only aerosol cans, so the model can't learn other hazardous waste such as batteries. Its real_world test scores are also unreliable, because the held-out set has only 12 Hazardous images.
- The real_world images are curated, so results show robustness to this shift rather than field performance in Phnom Penh.

## Development

```bash
make test
make lint
```
