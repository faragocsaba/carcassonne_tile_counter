# 🏰 Carcassonne BGA Board Analyzer & Tile Counter

An automated computer vision and deep learning pipeline designed to crop, classify, and count played Carcassonne board game tiles from Board Game Arena (BGA) screenshots. It calculates the remaining unplayed tile distribution based on the official 72-tile base set.

---

## 📌 Project Overview & Architecture

The system operates in 4 main stages:

1. **Asset Unpacking (`bga_asset_unpacker.py`)**: Slices BGA WebP spritesheet assets into individual base tiles and meeple PNGs.
2. **Synthetic Dataset Generation (`synthetic_dataset_generator.py`)**: Generates an augmented training dataset across 24 tile classes, accounting for rotations (0°, 90°, 180°, 270°) and standing/lying meeple placements in 5 player colors.
3. **Board Screenshot Cropping (`bga_tile_cropper.py`)**: Extracts played tiles from BGA screenshots using LAB chrominance background masking and 2D grid phase-aligned optimization.
4. **Tile Classification & Set Analysis (`carcassonne_set_analyzer.py`)**: Predicts tile categories using a fine-tuned **ResNet18** model and calculates the remaining unplayed tile distribution.

---

## 🛠️ Repository Structure

```text
.
├── carcassonne_bga_screenshots/   # Input folder for raw BGA screenshot JPGs
│   └── example_match.jpg
├── assets/                        # Unpacked BGA PNG assets
│   ├── tiles/                     # Base tile templates (tile_000.png to tile_028.png)
│   └── meeples/                   # Meeple overlays (5 colors, standing & lying)
├── generated_tiles/               # Synthetic dataset organized by tile code
├── extracted_tiles/               # Cropped tiles & reports per processed screenshot
├── bga_asset_unpacker.py          # Extracts PNG assets from raw BGA WebP sheets
├── synthetic_dataset_generator.py # Generates synthetic dataset with meeples & rotations
├── bga_tile_cropper.py            # Standalone grid cropper for BGA screenshots
├── train_tile_classifier.py       # Fine-tunes ResNet18 model on synthetic dataset
├── carcassonne_set_analyzer.py    # Main pipeline: Cropping + AI Prediction + Report
├── carcassonne_model.pth          # Trained PyTorch ResNet18 model weights
└── class_names.json               # Class index to tile code mapping
```

---

## 🚀 Quickstart Guide

### 1. Installation

Install the required Python dependencies:

```bash
pip install torch torchvision pillow opencv-python numpy
```

### 2. Asset Extraction & Dataset Generation

Place `tiles_first_edition.webp` and `meeples.webp` in the project root directory, then run:

```bash
# Extract raw tile & meeple PNGs into assets/
python bga_asset_unpacker.py

# Generate full synthetic training set in generated_tiles/
python synthetic_dataset_generator.py
```

### 3. Model Training

Fine-tune the ResNet18 classifier on the synthetic dataset:

```bash
python train_tile_classifier.py
```

This generates `carcassonne_model.pth` and `class_names.json`.

### 4. Board Screenshot Analysis

Place your BGA game screenshots (`.jpg` or `.jpeg`) into `carcassonne_bga_screenshots/` and run:

```bash
python carcassonne_set_analyzer.py
```

---

## 📊 Output Example

For each analyzed screenshot, a dedicated subdirectory is created in `extracted_tiles/` containing:
- `grid_detection_preview.jpg`: Verification image showing green grid bounding boxes around detected tiles.
- `debug_clean_board_mask.png`: Binary mask used for grid fitting.
- `analysis_report.txt`: Text summary comparing played vs. remaining counts.
- Subdirectories per tile category containing categorized tile crops.

### Sample Report (`analysis_report.txt`)

```text
===============================================================================
 CARCASSONNE TILE COUNT ANALYSIS REPORT
 Source Image: own_party1.jpg
===============================================================================
TILE_CODE    | PLAYED | SET_TOTAL | REMAINING
-------------------------------------------------
CCCCS        |      0 |         1 |         1
RRRR         |      0 |         1 |         1
CCCF         |      1 |         3 |         2
CCCFS        |      0 |         1 |         1
CCCR         |      0 |         1 |         1
CCCRS        |      0 |         2 |         2
RRRF         |      0 |         4 |         4
CFCF         |      0 |         1 |         1
CFCFS        |      0 |         2 |         2
RFRF         |      1 |         8 |         7
CCFF         |      0 |         3 |         3
CCFFS        |      0 |         2 |         2
CCRR         |      0 |         3 |         3
CCRRS        |      0 |         2 |         2
RRFF         |      1 |         9 |         8
CCFF2        |      0 |         2 |         2
CFCF2        |      1 |         3 |         2
RFFF         |      0 |         2 |         2
FFFF         |      0 |         4 |         4
CFFF         |      0 |         5 |         5
CRRF         |      0 |         3 |         3
CFRR         |      0 |         3 |         3
CRRR         |      0 |         3 |         3
CRFR         |      0 |         4 |         4
-------------------------------------------------
TOTAL LAND   |      4 |        72 |        68
===============================================================================
```

---

## 🎯 Tile Distribution (Base Game - 72 Tiles)

The analyzer uses the canonical base game tile counts:

| Code | Original Count | Description |
| :--- | :---: | :--- |
| `CCCCS` | 1 | Full city with shield |
| `RRRR` | 1 | 4-way road junction |
| `CCCF` / `CCCFS` | 3 / 1 | 3-sided city (without/with shield) |
| `CCCR` / `CCCRS` | 1 / 2 | 3-sided city + road (without/with shield) |
| `RRRF` | 4 | 3-way road junction |
| `CFCF` / `CFCFS` | 1 / 2 | Straight city segment (without/with shield) |
| `RFRF` | 8 | Straight road |
| `CCFF` / `CCFFS` | 3 / 2 | City corner (without/with shield) |
| `CCRR` / `CCRRS` | 3 / 2 | City corner + road curve (without/with shield) |
| `RRFF` | 9 | Curved road |
| `CCFF2` | 2 | Two separate city corners |
| `CFCF2` | 3 | Two separate straight city edges |
| `RFFF` / `FFFF` | 2 / 4 | Monastery (with/without road) |
| `CFFF` | 5 | City cap |
| `CRRF` / `CFRR` | 3 / 3 | City cap + road curve |
| `CRRR` | 3 | City cap + 3-way road junction |
| `CRFR` | 4 | City cap + straight road |
