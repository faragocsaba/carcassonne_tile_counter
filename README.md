# 🏰 Carcassonne BGA Board Analyzer & Tile Counter

An automated computer vision and deep learning pipeline designed to crop, classify, and count played Carcassonne board game tiles from Board Game Arena (BGA) screenshots. It calculates the remaining unplayed tile distribution based on the official 72-tile base set.

🚀 **Live Web App Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/fcsaba/carcassonne-board-analyzer)

---

## 📌 Project Overview & Architecture

The system operates in 4 main stages:

1. **Asset Unpacking (`bga_asset_unpacker.py`)**: Slices BGA WebP spritesheet assets into individual base tiles and meeple PNGs.
2. **Synthetic Dataset Generation (`synthetic_dataset_generator.py`)**: Generates an augmented training dataset across 24 tile classes, accounting for rotations (0°, 90°, 180°, 270°) and standing/lying meeple placements in 5 player colors.
3. **Board Screenshot Cropping (`bga_tile_cropper.py`)**: Extracts played tiles from BGA screenshots using LAB chrominance background masking and 2D grid phase-aligned optimization.
4. **Tile Classification & Set Analysis (`carcassonne_set_analyzer.py` / `app.py`)**: Predicts tile categories using a fine-tuned **ResNet18** model and calculates the remaining unplayed tile distribution. Interactive web interface powered by **Gradio** and deployed on **Hugging Face Spaces**.

---

## 🛠️ Repository Structure

```text
.
├── carcassonne_bga_screenshots/   # Input folder for raw BGA screenshot JPGs
│   └── example_match.jpg
├── assets/                        # Unpacked BGA PNG assets
│   ├── tiles/                     # Base tile templates (tile_000.png to tile_028.png)
│   └── meeples/                   # Meeple overlays (5 colors, standing & lying)
├── tiles/                         # Reference PNG icons per tile type (for HTML report UI)
├── generated_tiles/               # Synthetic dataset organized by tile code
├── extracted_tiles/               # Cropped tiles & reports per processed screenshot
├── bga_asset_unpacker.py          # Extracts PNG assets from raw BGA WebP sheets
├── synthetic_dataset_generator.py # Generates synthetic dataset with meeples & rotations
├── bga_tile_cropper.py            # Standalone grid cropper for BGA screenshots
├── train_tile_classifier.py       # Fine-tunes ResNet18 model on synthetic dataset
├── carcassonne_set_analyzer.py    # CLI batch pipeline: Cropping + AI Prediction + Report
├── app.py                         # Interactive Gradio Web App for Hugging Face Spaces
├── requirements.txt               # Dependencies for Hugging Face Spaces deployment
├── carcassonne_model.pth          # Trained PyTorch ResNet18 model weights
└── class_names.json               # Class index to tile code mapping
```

---

## 🌐 Web Application & Live Demo

Try the interactive web interface without installing anything:
👉 **[Carcassonne Board Analyzer on Hugging Face Spaces](https://huggingface.co/spaces/fcsaba/carcassonne-board-analyzer)**

To run the Gradio app locally:

```bash
pip install -r requirements.txt
python app.py
```

---

## 🚀 Quickstart Guide (CLI)

### 1. Installation

Install the required Python dependencies:

```bash
pip install torch torchvision pillow opencv-python numpy huggingface_hub gradio
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

The Gradio Web App (`app.py`) displays an interactive HTML table with embedded tile icons, played counts, canonical set totals, and remaining unplayed tile counts.

For batch processing via CLI (`carcassonne_set_analyzer.py`), a dedicated subdirectory is created in `extracted_tiles/` per screenshot containing:
- `grid_detection_preview.jpg`: Verification image showing green grid bounding boxes around detected tiles.
- `debug_clean_board_mask.png`: Binary mask used for grid fitting.
- `analysis_report.txt`: Text summary comparing played vs. remaining counts.
- Subdirectories per tile category containing categorized tile crops.

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
