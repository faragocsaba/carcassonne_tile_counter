"""
===============================================================================
Carcassonne BGA Board Screenshot Tile Cropper (bga_tile_cropper.py)
===============================================================================

Dependencies & Installation:
----------------------------
Install required Python libraries:
    pip install opencv-python numpy Pillow

Inputs & Directory Structure:
-----------------------------
Input directory: `carcassonne_bga_screenshots/`
- Place full Board Game Arena (BGA) game board screenshots (.jpg / .jpeg)
  into this folder.

Methodology & Processing Steps:
-------------------------------
1. Automatic Cleanup:
   - Purges and recreates the target `extracted_tiles/` output directory before
     batch processing to ensure clean builds.

2. LAB Chrominance Background & Placement Slot Masking (`get_real_tile_mask`):
   - Samples wood background color from image corners in LAB color space.
   - Filters out BOTH light wood background and BGA dark shaded placement slots
     by measuring 2D chrominance distance (a, b channels) relative to wood.
   - Explicitly preserves bright white roads, green fields, and player meeples.

3. Global Grid Phase & Pitch Optimization (`find_optimal_grid_parameters`):
   - Computes Canny boundary edges of the clean tile mask.
   - Optimizes 2D grid parameters: Tile Size S ∈ [80, 140] px and origin offsets
     x0, y0 ∈ [0, S-1] to perfectly align grid lines with tile outer borders.
   - Completely eliminates sub-tile splitting and grid offset shifts.

4. Tile Extraction & Output Normalization:
   - Evaluates tile coverage per grid cell (> 50% mask coverage required).
   - Crops valid tile bounding boxes and resizes them to standard 64x64px PNGs.
   - Retains verification debug images (`debug_clean_board_mask.png` and
     `grid_detection_preview.jpg`) per screenshot.

Outputs:
--------
Extracted tiles and debug overlays are saved into `extracted_tiles/`, organized
into subdirectories corresponding to each screenshot filename (without extension):

    extracted_tiles/
    ├── match_01/
    │   ├── tile_000.png
    │   ├── tile_001.png
    │   ├── debug_clean_board_mask.png
    │   └── grid_detection_preview.jpg
    └── ...
===============================================================================
"""

import os
import glob
import shutil
import cv2
import numpy as np
from PIL import Image

def get_real_tile_mask(img):
    """
    Creates a clean binary mask isolating actual played Carcassonne tiles.
    Uses LAB chrominance distance to unify light wood and BGA dark shaded slots
    into a single background class (0), while protecting authentic tile features.
    """
    h, w, _ = img.shape
    corner_size = 30

    # 1. Sample 4 corners for wood background color
    top_left = img[0:corner_size, 0:corner_size]
    top_right = img[0:corner_size, w - corner_size:w]
    bottom_left = img[h - corner_size:h, 0:corner_size]
    bottom_right = img[h - corner_size:h, w - corner_size:w]

    corner_pixels = np.vstack([
        top_left.reshape(-1, 3),
        top_right.reshape(-1, 3),
        bottom_left.reshape(-1, 3),
        bottom_right.reshape(-1, 3)
    ])

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    median_wood_bgr = np.median(corner_pixels, axis=0).astype(np.uint8)
    wood_lab = cv2.cvtColor(np.uint8([[median_wood_bgr]]), cv2.COLOR_BGR2LAB)[0, 0]

    # Distance to pure wood background in 3D LAB space
    dist_wood_3d = np.linalg.norm(img_lab.astype(np.float32) - wood_lab.astype(np.float32), axis=2)
    is_pure_wood = dist_wood_3d < 28.0

    # Chrominance distance in AB space (ignoring lightness L)
    chroma_dist = np.sqrt(
        (img_lab[:, :, 1].astype(np.float32) - wood_lab[1])**2 + 
        (img_lab[:, :, 2].astype(np.float32) - wood_lab[2])**2
    )
    L_channel = img_lab[:, :, 0]

    # Shaded placement slots: Chrominance close to wood, lightness L <= wood lightness + 5
    is_shaded_slot = (chroma_dist < 14.0) & (L_channel <= wood_lab[0] + 5)

    # Combined background mask
    is_bg = is_pure_wood | is_shaded_slot

    # 2. Explicit Protection for Real Tile Features in HSV Space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    is_white_road = (S <= 35) & (V >= 180)
    is_green_field = (H >= 25) & (H <= 90) & (S >= 25) & (V >= 25)
    is_black_meeple = (V < 25)

    # Real tile pixel mask
    is_tile_pixel = (~is_bg) | is_white_road | is_green_field | is_black_meeple

    raw_mask = (is_tile_pixel.astype(np.uint8)) * 255

    # Morphological cleaning
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

def process_single_bga_screenshot(image_path, target_subdir):
    """
    Processes a single BGA screenshot, crops valid 64x64px tiles using phase-aligned grid,
    and saves debug verification files.
    """
    print(f"\n[Processing Image] '{image_path}'...")

    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'")
        return

    overlay = img.copy()
    height, width, _ = img.shape

    os.makedirs(target_subdir, exist_ok=True)

    # 2. Generate Clean Real Tile Mask & Save Debug Image
    tile_mask = get_real_tile_mask(img)
    mask_debug_path = os.path.join(target_subdir, "debug_clean_board_mask.png")
    cv2.imwrite(mask_debug_path, tile_mask)

    # 3. Find Outer Pixel Extents
    tile_pixels = np.column_stack(np.where(tile_mask > 0))
    if len(tile_pixels) == 0:
        print(f"Error: No valid tile content detected in '{image_path}'. Check '{mask_debug_path}'.")
        return

    min_y, min_x = tile_pixels.min(axis=0)
    max_y, max_x = tile_pixels.max(axis=0)

    # 4. Find Optimal Grid Size S and Phase Offsets (x0, y0)
    tile_size, x0, y0 = find_optimal_grid_parameters(tile_mask)

    print(f"  └─ Fitted Tile Pitch S: {tile_size}px")
    print(f"  └─ Optimal Grid Phase Offset: (x0={x0}, y0={y0})")

    # Determine grid cell index ranges covering the played board area
    col_start = int(np.floor((min_x - x0) / tile_size))
    col_end = int(np.ceil((max_x - x0) / tile_size))
    row_start = int(np.floor((min_y - y0) / tile_size))
    row_end = int(np.ceil((max_y - y0) / tile_size))

    # 5. Crop Valid Tiles (Resized strictly to 64x64px)
    pil_img = Image.open(image_path)
    saved_count = 0

    for r in range(row_start, row_end + 1):
        y1 = y0 + r * tile_size
        y2 = y1 + tile_size

        if y1 < 0 or y2 > height:
            continue

        for c in range(col_start, col_end + 1):
            x1 = x0 + c * tile_size
            x2 = x1 + tile_size

            if x1 < 0 or x2 > width:
                continue

            cell_mask = tile_mask[y1:y2, x1:x2]
            if cell_mask.size == 0:
                continue

            fill_ratio = np.sum(cell_mask > 0) / cell_mask.size

            # Save cell only if predominantly real tile area (> 50%)
            if fill_ratio > 0.50:
                crop_box = (x1, y1, x2, y2)
                
                # Resized strictly to 64x64 pixels
                cropped_tile = pil_img.crop(crop_box).resize((64, 64), Image.Resampling.LANCZOS)

                filename = os.path.join(target_subdir, f"tile_{saved_count:03d}.png")
                cropped_tile.save(filename)

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                saved_count += 1

    # Save Grid Detection Preview Image
    preview_path = os.path.join(target_subdir, "grid_detection_preview.jpg")
    cv2.imwrite(preview_path, overlay)

    print(f"  └─ Extraction Complete: {saved_count} valid tiles (64x64px) saved to '{target_subdir}/'.")

def process_all_bga_screenshots(input_dir="carcassonne_bga_screenshots", output_dir="extracted_tiles"):
    """
    Finds all .jpg/.jpeg images in input_dir and crops tiles into subdirectories inside output_dir.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Purge output directory for clean builds
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"[Tile Cropper] Purged existing output directory '{output_dir}/'.")

    os.makedirs(output_dir, exist_ok=True)

    # Search for JPG / JPEG files
    image_paths = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg")) + 
        glob.glob(os.path.join(input_dir, "*.jpeg"))
    )

    if not image_paths:
        print(f"Warning: No JPG images found in '{input_dir}/'.")
        return

    print(f"[Tile Cropper] Found {len(image_paths)} screenshot(s) to process in '{input_dir}/'.")

    for image_path in image_paths:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        target_subdir = os.path.join(output_dir, base_name)

        process_single_bga_screenshot(image_path, target_subdir)

    print(f"\n[Tile Cropper] 🎉 Batch Processing Finished! All outputs saved to '{output_dir}/'.")

if __name__ == "__main__":
    process_all_bga_screenshots(
        input_dir="carcassonne_bga_screenshots",
        output_dir="extracted_tiles"
    )