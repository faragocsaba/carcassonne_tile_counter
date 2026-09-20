# 🏰 Carcassonne BGA Board Analyzer & Tile Counting AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Computer Vision & Deep Learning pipeline that automatically detects, crops, classifies, and counts played and remaining tiles from **Board Game Arena (BGA)** Carcassonne game board screenshots.

---

## 🌟 Key Features

* **Synthetic Data Generator (`generate_tiles_dataset.py`)**: Programmatically renders thousands of synthetic Carcassonne tile image variations (clean, rotated, and meeple-covered) to overcome real-world dataset scarcity.
* **Phase-Aligned Grid Cropper (`bga_tile_cropper.py`)**: Uses LAB/HSV color masking and Canny edge 2D phase-offset optimization to precisely isolate played tiles from wood backgrounds and BGA shaded placement slots.
* **Fine-Tuned ResNet18 Classifier (`train_tile_classifier.py`)**: Leverages transfer learning with strong data augmentation (RandomAffine, ColorJitter, RandomErasing) to achieve near-perfect classification accuracy across all 24 canonical tile classes.
* **Remaining Set Calculator (`carcassonne_set_analyzer.py`)**: Analyzes full board screenshots and produces detailed breakdown reports comparing played tiles against the official 72-tile base set distribution.

---

## 📐 Architecture & Pipeline Overview

```
[ Full BGA Screenshot ] 
       │
       ▼
[ OpenCV Masking & Phase Grid Cropper ] ──► (Extracted 64x64 Tile PNGs)
       │
       ▼
[ Pretrained ResNet18 CNN Classifier ] ──► (Predicted Class: CCCR, RRFF, etc.)
       │
       ▼
[ Canonical 72-Tile Tally Engine ]    ──► (Detailed Remaining Set Report TXT)
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites & Setup
Clone the repository and install required Python packages:

```bash
git clone https://github.com/your-username/carcassonne-bga-analyzer.git
cd carcassonne-bga-analyzer
pip install torch torchvision pillow opencv-python numpy
```

### 2. Dataset Generation (Optional)
Generate the synthetic training dataset (~2,000 images across 24 tile types):

```bash
python generate_tiles_dataset.py
```

### 3. Model Training
Train/fine-tune the ResNet18 tile classifier on the synthetic dataset:

```bash
python train_tile_classifier.py
```
*Outputs `carcassonne_model.pth` and `class_names.json`.*

### 4. Board Analysis
Place full BGA screenshot JPGs into the `carcassonne_bga_screenshots/` directory and run:

```bash
python carcassonne_set_analyzer.py
```

Check the results and generated text reports in `extracted_tiles/<screenshot_name>/`.

---

## 📊 Sample Analysis Output

```text
===============================================================================
 CARCASSONNE TILE COUNT ANALYSIS REPORT
 Source Image: match_screenshot_01.jpg
===============================================================================
TILE_CODE    | PLAYED | SET_TOTAL | REMAINING
-----------------------------------------------
CCCCS        |      1 |         1 |         0
RRRR         |      0 |         1 |         1
CCCF         |      2 |         3 |         1
CCCFS        |      1 |         1 |         0
CCCR         |      1 |         1 |         0
...
-----------------------------------------------
TOTAL LAND   |     24 |        72 |        48
===============================================================================
```

---

## 📁 Repository Structure

```
├── carcassonne_bga_screenshots/ # Input directory for full board screenshots
├── generated_tiles/             # Synthetic dataset output directory
├── extracted_tiles/             # Cropped tiles, debug masks, and analysis reports
├── generate_tiles_dataset.py    # Synthetic image generation pipeline
├── train_tile_classifier.py     # PyTorch ResNet18 fine-tuning script
├── predict_tiles.py             # Inference script for isolated tiles
├── carcassonne_set_analyzer.py  # Main pipeline & remaining tile calculator
├── class_names.json             # Model index-to-class mapping
├── carcassonne_model.pth        # Trained PyTorch model weights
└── README.md                    # Project documentation
```

---

## 🤝 Acknowledgments & License

* Board game artwork inspired by **Carcassonne** (designed by Klaus-Jürgen Wrede, published by Hans im Glück).
* Board game screenshot captures courtesy of **Board Game Arena (BGA)**.
* Distributed under the **MIT License**.
