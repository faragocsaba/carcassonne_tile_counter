"""
===============================================================================
Carcassonne Board Batch Analyzer (carcassonne_set_analyzer.py)
===============================================================================

Dependencies & Installation:
----------------------------
pip install torch torchvision pillow opencv-python numpy

Prerequisites:
--------------
1. A trained model file `carcassonne_model.pth` in the root directory.
2. The `class_names.json` mapping file in the root directory.
3. Screenshots placed inside `carcassonne_bga_screenshots/*.jpg`

Workflow & Batch Processing Steps:
----------------------------------
1. Directory Cleanup:
   - Purges existing target subdirectories under `extracted_tiles/` to ensure
     clean builds.

2. Batch Processing Loop:
   - Iterates through all `.jpg` / `.jpeg` files in `carcassonne_bga_screenshots/`.
   - For each file (e.g., `carcassonne_bga_screenshot_2.jpg`), creates a dedicated
     output directory `extracted_tiles/carcassonne_bga_screenshot_2/`.

3. Tile Cropping & Phase-Aligned Grid Alignment:
   - Crops played tiles from screenshot and saves debug files
     (`debug_clean_board_mask.png` and `grid_detection_preview.jpg`).

4. ResNet18 Tile Prediction & Categorization:
   - Classifies each 64x64px tile using the trained ResNet18 model.
   - Saves classified tiles into predicted category subfolders within each screenshot's
     output directory.

5. Report Generation & Text Export:
   - Compares played tile counts against the OFFICIAL 72-tile base game distribution.
   - Outputs a detailed text report to both the console and a text file named
     `analysis_report.txt` inside the corresponding `extracted_tiles/` subdirectory.

Outputs:
--------
    extracted_tiles/
    ├── carcassonne_bga_screenshot_1/
    │   ├── analysis_report.txt
    │   ├── debug_clean_board_mask.png
    │   ├── grid_detection_preview.jpg
    │   ├── CCCR/
    │   ├── RRFF/
    │   └── ...
    └── carcassonne_bga_screenshot_2/
        └── ...
===============================================================================
"""

import os
import glob
import shutil
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import cv2
import numpy as np
from collections import Counter

# ==============================================================================
# Configuration & Official Canonical Distribution (Base Set, 72 Tiles)
# ==============================================================================
MODEL_PATH = "carcassonne_model.pth"
CLASSES_PATH = "class_names.json"
INPUT_DIR = "carcassonne_bga_screenshots"
OUTPUT_DIR = "extracted_tiles"

# OFFICIAL Carcassonne 1st Edition Base Set distribution (Sum = 72 land tiles)
# Corrected based on user input.
ORIGINAL_SET_COMPOSITION = {
    "CCCCS": 1, "RRRR": 1, "CCCF": 3, "CCCFS": 1,  "CCCR": 1, "CCCRS": 2,
    "RRRF": 4, "CFCF": 1, "CFCFS": 2, "RFRF": 8, "CCFF": 3, "CCFFS": 2,
    "CCRR": 3, "CCRRS": 2, "RRFF": 9, "CCFF2": 2, "CFCF2": 3, "RFFF": 2,
    "FFFF": 4, "CFFF": 5, "CRRF": 3, "CFRR": 3, "CRRR": 3, "CRFR": 4,
}

IMAGE_SIZE = (64, 64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Prediction Helpers
# ==============================================================================
def get_resnet_model(num_classes):
    """Defines ResNet18 architecture matching trained weights."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

class TileClassifier:
    def __init__(self, model_path, classes_path, device):
        self.device = device
        
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Missing required class mapping file: '{classes_path}'")
        with open(classes_path, "r") as f:
            self.idx_to_class = json.load(f)
        self.num_classes = len(self.idx_to_class)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing trained model file: '{model_path}'")
        self.model = get_resnet_model(self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, pil_img):
        """Classifies a single PIL image and returns label + confidence score."""
        img_rgb = pil_img.convert("RGB")
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probabilities, 1)

        predicted_label = self.idx_to_class[str(pred_idx.item())]
        confidence_score = conf.item() * 100
        return predicted_label, confidence_score

# ==============================================================================
# 2. Tile Cropper Helpers
# ==============================================================================
def get_real_tile_mask(img):
    """Creates a binary mask isolating played tiles from wood and shaded slots."""
    h, w, _ = img.shape
    corner_size = 30
    top_left = img[0:corner_size, 0:corner_size]
    top_right = img[0:corner_size, w - corner_size:w]
    bottom_left = img[h - corner_size:h, 0:corner_size]
    bottom_right = img[h - corner_size:h, w - corner_size:w]
    corner_pixels = np.vstack([top_left.reshape(-1, 3), top_right.reshape(-1, 3), bottom_left.reshape(-1, 3), bottom_right.reshape(-1, 3)])
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    median_wood_bgr = np.median(corner_pixels, axis=0).astype(np.uint8)
    wood_lab = cv2.cvtColor(np.uint8([[median_wood_bgr]]), cv2.COLOR_BGR2LAB)[0, 0]
    
    dist_wood_3d = np.linalg.norm(img_lab.astype(np.float32) - wood_lab.astype(np.float32), axis=2)
    is_pure_wood = dist_wood_3d < 28.0
    
    chroma_dist = np.sqrt((img_lab[:, :, 1].astype(np.float32) - wood_lab[1])**2 + (img_lab[:, :, 2].astype(np.float32) - wood_lab[2])**2)
    L_channel = img_lab[:, :, 0]
    is_shaded_slot = (chroma_dist < 14.0) & (L_channel <= wood_lab[0] + 5)
    
    is_bg = is_pure_wood | is_shaded_slot
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    is_white_road = (S <= 35) & (V >= 180)
    is_green_field = (H >= 25) & (H <= 90) & (S >= 25) & (V >= 25)
    is_black_meeple = (V < 25)
    
    is_tile_pixel = (~is_bg) | is_white_road | is_green_field | is_black_meeple
    raw_mask = (is_tile_pixel.astype(np.uint8)) * 255
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    return clean_mask

def find_optimal_grid_parameters(tile_mask):
    """
    Optimizes 2D grid parameters (Tile Size S, Offset x0, Offset y0)
    by aligning grid lines with the outer boundary edges of played tiles.
    """
    h, w = tile_mask.shape
    
    # Compute boundary edges of the clean tile mask
    mask_edges = cv2.Canny(tile_mask, 100, 200)
    
    v_proj = np.sum(mask_edges, axis=0)
    h_proj = np.sum(mask_edges, axis=1)

    scores_by_S = {}

    # 1. Expanded search pitch range for individual tiles (35px to 110px)
    for S in range(35, 110):
        # Best x0 offset for candidate S
        best_x_score = -1
        best_x_offset = 0
        for x0 in range(S):
            x_indices = np.arange(x0, w, S)
            if len(x_indices) > 0:
                # Use np.mean to prevent bias caused by varying number of grid lines
                score_x = np.mean(v_proj[x_indices])
                if score_x > best_x_score:
                    best_x_score = score_x
                    best_x_offset = x0

        # Best y0 offset for candidate S
        best_y_score = -1
        best_y_offset = 0
        for y0 in range(S):
            y_indices = np.arange(y0, h, S)
            if len(y_indices) > 0:
                score_y = np.mean(h_proj[y_indices])
                if score_y > best_y_score:
                    best_y_score = score_y
                    best_y_offset = y0

        total_score = best_x_score + best_y_score
        scores_by_S[S] = (total_score, best_x_offset, best_y_offset)

    # 2. Find maximum achieved score across all tile sizes
    max_score = max(data[0] for data in scores_by_S.values())

    # 3. Fundamental frequency selection: choose smallest tile size S
    # achieving at least 85% of the peak score.
    # Prevents selecting 2x or 3x harmonic multiples.
    candidate_S_list = [S for S, (score, _, _) in scores_by_S.items() if score >= 0.85 * max_score]
    best_S = min(candidate_S_list)
    
    _, best_x0, best_y0 = scores_by_S[best_S]

    return best_S, best_x0, best_y0

def detect_and_crop_screenshot(img_path, target_subdir):
    """Slices tiles from screenshot, saves debug images in target_subdir, returns cropped PIL tiles."""
    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None:
        raise FileNotFoundError(f"Could not load image: '{img_path}'")
    height, width, _ = img_cv2.shape
    overlay = img_cv2.copy()
    
    # Save clean mask
    tile_mask = get_real_tile_mask(img_cv2)
    cv2.imwrite(os.path.join(target_subdir, "debug_clean_board_mask.png"), tile_mask)

    tile_pixels = np.column_stack(np.where(tile_mask > 0))
    if len(tile_pixels) == 0:
        return []

    min_y, min_x = tile_pixels.min(axis=0)
    max_y, max_x = tile_pixels.max(axis=0)
    
    tile_size, x0, y0 = find_optimal_grid_parameters(tile_mask)
    
    col_start = int(np.floor((min_x - x0) / tile_size))
    col_end = int(np.ceil((max_x - x0) / tile_size))
    row_start = int(np.floor((min_y - y0) / tile_size))
    row_end = int(np.ceil((max_y - y0) / tile_size))
    
    pil_img_src = Image.open(img_path)
    cropped_pil_tiles = []
    
    for r in range(row_start, row_end + 1):
        y1 = y0 + r * tile_size
        y2 = y1 + tile_size
        if y1 < 0 or y2 > height: continue
        
        for c in range(col_start, col_end + 1):
            x1 = x0 + c * tile_size
            x2 = x1 + tile_size
            if x1 < 0 or x2 > width: continue
            
            cell_mask = tile_mask[y1:y2, x1:x2]
            if cell_mask.size == 0: continue
            
            fill_ratio = np.sum(cell_mask > 0) / cell_mask.size
            if fill_ratio > 0.50:
                crop_box = (x1, y1, x2, y2)
                tile_crop = pil_img_src.crop(crop_box).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                cropped_pil_tiles.append(tile_crop)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
    # Save grid detection preview
    cv2.imwrite(os.path.join(target_subdir, "grid_detection_preview.jpg"), overlay)
    return cropped_pil_tiles

# ==============================================================================
# 3. Report Builder
# ==============================================================================
def build_report_text(image_filename, played_counts):
    """Formats played vs original vs remaining counts into a clean text table."""
    lines = []
    lines.append("===============================================================================")
    lines.append(f" CARCASSONNE TILE COUNT ANALYSIS REPORT")
    lines.append(f" Source Image: {image_filename}")
    lines.append("===============================================================================")
    lines.append(f"{'TILE_CODE':12s} | {'PLAYED':>6s} | {'SET_TOTAL':>9s} | {'REMAINING':>9s}")
    lines.append("-" * 49)

    total_original = sum(ORIGINAL_SET_COMPOSITION.values())
    total_played = 0
    total_remaining = total_original

    # Iterate based on the original set composition order
    for tile_code, original_count in ORIGINAL_SET_COMPOSITION.items():
        played_count = played_counts.get(tile_code, 0)
        
        # In case prediction finds a tile not in the canonical set (should not happen with good model)
        remaining_count = original_count - played_count

        total_played += played_count
        total_remaining -= played_count

        lines.append(f"{tile_code:12s} | {played_count:>6d} | {original_count:>9d} | {remaining_count:>9d}")

    lines.append("-" * 49)
    lines.append(f"{'TOTAL LAND':12s} | {total_played:>6d} | {total_original:>9d} | {total_remaining:>9d}")
    lines.append("===============================================================================")
    return "\n".join(lines)

# ==============================================================================
# 4. Main Batch Processing Engine
# ==============================================================================
def process_all_screenshots():
    """Batch processes all screenshots in input directory and generates evaluation reports."""
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return

    # Find all JPG/JPEG files
    image_paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + 
        glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
    )

    if not image_paths:
        print(f"Warning: No JPG images found in '{INPUT_DIR}/'.")
        return

    print(f"\n===============================================================================")
    print(f"[Board Analyzer] Found {len(image_paths)} screenshot(s) in '{INPUT_DIR}/'.")
    print(f"===============================================================================")

    # Initialize Tile Classifier Model
    try:
        classifier = TileClassifier(MODEL_PATH, CLASSES_PATH, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        target_subdir = os.path.join(OUTPUT_DIR, base_name)

        print(f"\n[Processing] '{image_path}' -> '{target_subdir}/'...")

        # Purge existing output directory for clean build
        if os.path.exists(target_subdir):
            shutil.rmtree(target_subdir)
            print(f"  └─ Purged existing directory '{target_subdir}/'.")

        os.makedirs(target_subdir, exist_ok=True)

        # 1. Slice tiles and save debug images
        cropped_pil_tiles = detect_and_crop_screenshot(image_path, target_subdir)
        if not cropped_pil_tiles:
            print(f"  └─ Warning: No valid tiles detected in '{image_path}'. Skipping.")
            continue

        # 2. Classify tiles & organize into predicted subfolders
        played_counts = Counter()
        for idx, pil_tile in enumerate(cropped_pil_tiles):
            label, confidence = classifier.predict_image(pil_tile)
            played_counts[label] += 1

            # Save tile into predicted subfolder for verification
            cat_dir = os.path.join(target_subdir, label)
            os.makedirs(cat_dir, exist_ok=True)
            fname = f"{confidence:.0f}pct_tile_{idx:03d}.png"
            pil_tile.save(os.path.join(cat_dir, fname))

        # 3. Generate & display text report
        report_text = build_report_text(os.path.basename(image_path), played_counts)
        print("\n" + report_text)

        # 4. Save report text file
        report_path = os.path.join(target_subdir, "analysis_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"\n  └─ Analysis report saved to '{report_path}'.")

    print(f"\n===============================================================================")
    print(f"[Board Analyzer] 🎉 All screenshots processed! Results saved to '{OUTPUT_DIR}/'.")
    print(f"===============================================================================")

if __name__ == "__main__":
    process_all_screenshots()