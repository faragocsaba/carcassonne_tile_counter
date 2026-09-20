"""
===============================================================================
Carcassonne BGA Synthetic Tile Dataset Generator (generate_tiles_dataset.py)
===============================================================================

Dependencies & Installation:
----------------------------
Install required Python libraries:
    pip install Pillow

Prerequisites & Inputs:
-----------------------
Before running this script, `bga_asset_unpacker.py` must be executed to extract
and prepare the necessary PNG assets in the following structure:

1. assets/tiles/
   - The 29 base tile source images of Carcassonne (tile_000.png - tile_028.png)
   - Format: 64x64 pixels, RGBA PNG

2. assets/meeples/
   - Meeple figures in 5 colors (blue, green, black, red, yellow)
   - In two poses: standing (knight/thief/monk) and lying (farmer)
   - File names: meeple_standing_<color>.png, meeple_lying_<color>.png
   - Format: RGBA PNG with transparent background

Workflow & Usage:
-----------------
Run from the command line:
    python generate_tiles_dataset.py

Execution Steps:
  1. Cleans and purges any existing `generated_tiles/` output directory.
  2. Iterates through the configurations of all 29 tile source files (TILES_CONFIG).
  3. Generates 4 rotation variants (0°, 90°, 180°, 270°) for each tile.
  4. Saves clean base tiles (without meeples).
  5. Overlays standing and lying meeple figures at calibrated anchor coordinates
     in all 5 colors, using native BGA 24x24 pixel scaling.

Outputs:
--------
The synthetic dataset is generated into `generated_tiles/`, organized into 24
unique Board Game Arena tile category directories:

    generated_tiles/
    ├── CCCR/
    │   ├── CCCR_000_v000_clean.png
    │   ├── CCCR_000_v000_city_standing_blue.png
    │   ├── CCCR_090_v000_field_left_lying_red.png
    │   └── ...
    ├── RRFF/
    ├── FFFF/
    └── ...

Naming Convention:
    <TILECODE>_<ROTATION>_<VARIANT>_<ANCHOR>_<POSE>_<COLOR>.png
    Example: RRFF_090_v018_road_standing_black.png
===============================================================================
"""

import os
import shutil
from PIL import Image

def rotate_coordinate(x, y, angle, image_size=64):
    """
    Transforms a 2D coordinate for 0, 90, 180, and 270 degree clockwise rotations.
    """
    max_idx = image_size - 1
    if angle == 0:
        return x, y
    elif angle == 90:   # 90 deg clockwise
        return max_idx - y, x
    elif angle == 180: # 180 deg
        return max_idx - x, max_idx - y
    elif angle == 270: # 270 deg clockwise
        return y, max_idx - x
    return x, y

TILES_CONFIG = {
    "tile_000.png": {
        "tile_code": "CCCR",
        "description": "3-sided city + bottom road (no shield)",
        "anchors": [
            ("city", "standing", (32, 22)),
            ("road", "standing", (32, 52)),
            ("field_left", "lying", (18, 55)),
            ("field_right", "lying", (46, 55))
        ]
    },
    "tile_001.png": {
        "tile_code": "CCCRS",
        "description": "3-sided city + bottom road (with shield)",
        "anchors": [
            ("city", "standing", (32, 22)),
            ("road", "standing", (32, 52)),
            ("field_left", "lying", (18, 55)),
            ("field_right", "lying", (46, 55))
        ]
    },
    "tile_002.png": {
        "tile_code": "CCFF",
        "description": "City corner CCFF (no shield)",
        "anchors": [
            ("city", "standing", (14, 14)),
            ("field", "lying", (46, 46))
        ]
    },
    "tile_003.png": {
        "tile_code": "CCFFS",
        "description": "City corner CCFF (with shield)",
        "anchors": [
            ("city", "standing", (14, 14)),
            ("field", "lying", (46, 46))
        ]
    },
    "tile_004.png": {
        "tile_code": "CCRR",
        "description": "City corner + road curve (no shield)",
        "anchors": [
            ("city", "standing", (16, 16)),
            ("field_inner", "lying", (34, 34)),
            ("road", "standing", (43, 43)),
            ("field_outer", "lying", (54, 54))
        ]
    },
    "tile_005.png": {
        "tile_code": "CCRRS",
        "description": "City corner + road curve (with shield)",
        "anchors": [
            ("city", "standing", (16, 16)),
            ("field_inner", "lying", (34, 34)),
            ("road", "standing", (43, 43)),
            ("field_outer", "lying", (54, 54))
        ]
    },
    "tile_006.png": {
        "tile_code": "CFCF",
        "description": "City straight segment CFCF",
        "anchors": [
            ("city", "standing", (32, 32)),
            ("field_left", "lying", (10, 32)),
            ("field_right", "lying", (54, 32))
        ]
    },
    "tile_007.png": {
        "tile_code": "CFCFS",
        "description": "City straight segment CFCFS",
        "anchors": [
            ("city", "standing", (32, 32)),
            ("field_left", "lying", (10, 32)),
            ("field_right", "lying", (54, 32))
        ]
    },
    "tile_008.png": {
        "tile_code": "CCFF2",
        "description": "Two separate city segments CCFF2 (top and left)",
        "anchors": [
            ("city_top", "standing", (32, 10)),
            ("city_left", "standing", (10, 32)),
            ("field", "lying", (40, 40))
        ]
    },
    "tile_009.png": {
        "tile_code": "CFCF2",
        "description": "Two separate city segments CFCF2 (top and bottom)",
        "anchors": [
            ("city_top", "standing", (32, 10)),
            ("city_bottom", "standing", (32, 54)),
            ("field", "lying", (32, 32))
        ]
    },
    "tile_010.png": {
        "tile_code": "CFFF",
        "description": "City cap CFFF (variant 1)",
        "anchors": [
            ("city", "standing", (32, 10)),
            ("field", "lying", (32, 38))
        ]
    },
    "tile_011.png": {
        "tile_code": "CFFF",
        "description": "City cap CFFF (variant 2)",
        "anchors": [
            ("city", "standing", (32, 10)),
            ("field", "lying", (32, 38))
        ]
    },
    "tile_012.png": {
        "tile_code": "CFRR",
        "description": "City cap + left-bottom road curve",
        "anchors": [
            ("city", "standing", (32, 10)),
            ("road", "standing", (25, 38)),
            ("field_inner", "lying", (39, 32)),
            ("field_outer", "lying", (13, 50))
        ]
    },
    "tile_013.png": {
        "tile_code": "CRRF",
        "description": "City cap + bottom-right road curve",
        "anchors": [
            ("city", "standing", (32, 10)),
            ("road", "standing", (38, 38)),
            ("field_inner", "lying", (24, 32)),
            ("field_outer", "lying", (50, 50))
        ]
    },
    "tile_014.png": {
        "tile_code": "CRRR",
        "description": "City cap + 3-way road junction",
        "anchors": [
            ("city", "standing", (32, 10)),
            ("road_left", "standing", (14, 38)),
            ("road_right", "standing", (50, 38)),
            ("road_bottom", "standing", (32, 54)),
            ("field_top", "lying", (32, 22)),
            ("field_bottom_left", "lying", (12, 52)),
            ("field_bottom_right", "lying", (52, 52))
        ]
    },
    "tile_015.png": {
        "tile_code": "CRFR",
        "description": "City cap + horizontal straight road",
        "anchors": [
            ("city", "standing", (32, 10)),
            ("field_top", "lying", (32, 22)),
            ("road", "standing", (32, 33)),
            ("field_bottom", "lying", (32, 50))
        ]
    },
    "tile_016.png": {
        "tile_code": "RFRF",
        "description": "Straight vertical road RFRF (variant 1)",
        "anchors": [
            ("road", "standing", (32, 32)),
            ("field_left", "lying", (14, 32)),
            ("field_right", "lying", (50, 32))
        ]
    },
    "tile_017.png": {
        "tile_code": "RFRF",
        "description": "Straight vertical road RFRF (variant 2)",
        "anchors": [
            ("road", "standing", (32, 32)),
            ("field_left", "lying", (14, 32)),
            ("field_right", "lying", (50, 32))
        ]
    },
    "tile_018.png": {
        "tile_code": "RRFF",
        "description": "Road curve RRFF (variant 1)",
        "anchors": [
            ("road", "standing", (29, 33)),
            ("field_inner", "lying", (13, 50)),
            ("field_outer", "lying", (48, 16))
        ]
    },
    "tile_019.png": {
        "tile_code": "RRFF",
        "description": "Road curve RRFF (variant 2)",
        "anchors": [
            ("road", "standing", (29, 33)),
            ("field_inner", "lying", (13, 50)),
            ("field_outer", "lying", (48, 16))
        ]
    },
    "tile_020.png": {
        "tile_code": "RRFF",
        "description": "Road curve RRFF (variant 3)",
        "anchors": [
            ("road", "standing", (29, 33)),
            ("field_inner", "lying", (13, 50)),
            ("field_outer", "lying", (48, 16))
        ]
    },
    "tile_021.png": {
        "tile_code": "RRRF",
        "description": "3-way road junction RRRF (variant 1)",
        "anchors": [
            ("road_left", "standing", (14, 29)),
            ("road_right", "standing", (50, 30)),
            ("road_bottom", "standing", (32, 52)),
            ("field_top", "lying", (32, 14)),
            ("field_bottom_left", "lying", (13, 50)),
            ("field_bottom_right", "lying", (51, 50))
        ]
    },
    "tile_022.png": {
        "tile_code": "RRRF",
        "description": "3-way road junction RRRF (variant 2)",
        "anchors": [
            ("road_left", "standing", (14, 29)),
            ("road_right", "standing", (50, 30)),
            ("road_bottom", "standing", (32, 52)),
            ("field_top", "lying", (32, 14)),
            ("field_bottom_left", "lying", (13, 50)),
            ("field_bottom_right", "lying", (51, 50))
        ]
    },
    "tile_023.png": {
        "tile_code": "RRRR",
        "description": "4-way road junction RRRR",
        "anchors": [
            ("road_top", "standing", (32, 12)),
            ("road_bottom", "standing", (32, 52)),
            ("road_left", "standing", (12, 32)),
            ("road_right", "standing", (52, 32)),
            ("field_top_left", "lying", (14, 14)),
            ("field_top_right", "lying", (50, 14)),
            ("field_bottom_left", "lying", (14, 50)),
            ("field_bottom_right", "lying", (50, 50))
        ]
    },
    "tile_024.png": {
        "tile_code": "FFFF",
        "description": "Monastery without road FFFF",
        "anchors": [
            ("monastery", "standing", (32, 32)),
            ("field", "lying", (53, 12))
        ]
    },
    "tile_025.png": {
        "tile_code": "RFFF",
        "description": "Monastery with road RFFF",
        "anchors": [
            ("monastery", "standing", (32, 28)),
            ("road", "standing", (32, 52)),
            ("field", "lying", (53, 12))
        ]
    },
    "tile_026.png": {
        "tile_code": "CCCCS",
        "description": "Full city tile with shield CCCCS",
        "anchors": [
            ("city", "standing", (32, 32))
        ]
    },
    "tile_027.png": {
        "tile_code": "CCCF",
        "description": "3-sided city without road CCCF (no shield)",
        "anchors": [
            ("city", "standing", (32, 22)),
            ("field", "lying", (32, 52))
        ]
    },
    "tile_028.png": {
        "tile_code": "CCCFS",
        "description": "3-sided city without road CCCFS (with shield)",
        "anchors": [
            ("city", "standing", (32, 22)),
            ("field", "lying", (32, 52))
        ]
    }
}

def generate_batch_dataset(tiles_dir="assets/tiles", meeples_dir="assets/meeples", output_dir="generated_tiles"):
    """
    Generates full Carcassonne tile dataset structured by tile codes (24 unique categories).
    Uses full 24x24 BGA native meeple scaling.
    Purges target folder prior to generation.
    """
    if not os.path.exists(tiles_dir) or not os.path.exists(meeples_dir):
        print("Error: Missing asset directories. Ensure 'assets/tiles/' and 'assets/meeples/' exist.")
        return

    # Automatic directory purge
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"[Batch Generator] Purged existing directory '{output_dir}/'.")

    colors = ["blue", "green", "black", "red", "yellow"]
    rotations = [0, 90, 180, 270]
    
    # Native BGA 24x24px scale
    meeple_size = (24, 24)
    total_generated = 0

    print(f"[Batch Generator] Generating COMPLETE dataset into clean '{output_dir}/'...")

    for tile_file, config in TILES_CONFIG.items():
        tile_path = os.path.join(tiles_dir, tile_file)
        if not os.path.exists(tile_path):
            print(f"Warning: File '{tile_file}' not found. Skipping.")
            continue

        base_tile = Image.open(tile_path).convert("RGBA")
        tile_code = config["tile_code"]
        anchors = config["anchors"]
        
        var_id = f"v{tile_file.split('.')[0].split('_')[1]}"

        tile_dir = os.path.join(output_dir, tile_code)
        os.makedirs(tile_dir, exist_ok=True)

        for angle in rotations:
            angle_str = f"{angle:03d}"

            # 1. Rotate base tile
            rotated_tile = base_tile.rotate(-angle, resample=Image.Resampling.LANCZOS)

            # 2. Save CLEAN image
            clean_filename = f"{tile_code}_{angle_str}_{var_id}_clean.png"
            rotated_tile.convert("RGB").save(os.path.join(tile_dir, clean_filename))
            total_generated += 1

            # 3. Generate meeple overlays
            for anchor_name, pose, (raw_x, raw_y) in anchors:
                rot_x, rot_y = rotate_coordinate(raw_x, raw_y, angle, image_size=64)

                for color in colors:
                    meeple_filename = f"meeple_{pose}_{color}.png"
                    meeple_path = os.path.join(meeples_dir, meeple_filename)

                    if not os.path.exists(meeple_path):
                        continue

                    meeple_img = Image.open(meeple_path).convert("RGBA")
                    meeple_scaled = meeple_img.resize(meeple_size, Image.Resampling.LANCZOS)

                    composite = rotated_tile.copy()
                    paste_x = rot_x - meeple_scaled.width // 2
                    paste_y = rot_y - meeple_scaled.height // 2

                    composite.paste(meeple_scaled, (paste_x, paste_y), meeple_scaled)

                    out_filename = f"{tile_code}_{angle_str}_{var_id}_{anchor_name}_{pose}_{color}.png"
                    composite.convert("RGB").save(os.path.join(tile_dir, out_filename))
                    total_generated += 1

    print(f"\n[Batch Generator] 🎉 Generation Complete!")
    print(f"[Batch Generator] Total generated dataset size: {total_generated} images across all 29 tile sources (000-028).")

if __name__ == "__main__":
    generate_batch_dataset()