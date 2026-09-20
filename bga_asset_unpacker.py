"""
===============================================================================
Carcassonne BGA Asset Unpacker (bga_asset_unpacker.py)
===============================================================================

Dependencies & Installation:
----------------------------
Install required Python libraries:
    pip install Pillow

How to Obtain Raw Assets from Board Game Arena (BGA):
------------------------------------------------------
1. Open https://boardgamearena.com in your web browser and enter any active
   or completed Carcassonne match (or game replay).
2. Open Browser Developer Tools by pressing F12 (or Ctrl+Shift+I / Cmd+Option+I).
3. Navigate to the "Network" tab.
4. Set the network filter to "Img" (or type "webp" into the search/filter bar).
5. Refresh the page (F5) if image network requests are already cached and not listed.
6. Locate the two required WebP spritesheet files in the network request list:
   - Tile Spritesheet: Look for `tiles_first_edition.webp` (or the 1st edition tile sheet).
   - Meeple Spritesheet: Look for `meeples.webp` (or the meeple sprite sheet).
7. Right-click each file -> "Open in new tab" -> Save Image As... (Ctrl+S).
8. Save both files directly into the root directory of this project with the exact names:
   - `tiles_first_edition.webp`
   - `meeples.webp`

Inputs:
-------
- `tiles_first_edition.webp`: WebP spritesheet containing base game tile textures.
- `meeples.webp`: WebP spritesheet containing player meeple textures.

Workflow & Slicing Logic:
-------------------------
1. Automatic Cleanup:
   - Automatically purges existing target folders (`assets/tiles/` and `assets/meeples/`)
     before processing to ensure a clean build environment.

2. `unpack_tile_spritesheet()`:
   - Slices the WebP tile sheet using a 23-column by 4-row grid alignment.
   - Extracts strictly the first 29 base game tiles (tile_000 to tile_028).
   - Resizes each cropped tile to a standardized 64x64px RGBA PNG image.

3. `unpack_meeple_spritesheet()`:
   - Slices the WebP meeple sheet using a 5-column by 6-row grid layout.
   - Crops the first 2 rows (Row 0: Standing meeples, Row 1: Lying farmer meeples).
   - Extracts 10 sprite variants across 5 player colors (blue, green, black, red, yellow).

Outputs:
--------
The extracted individual PNG assets are saved into the `assets/` directory:

    assets/
    ├── tiles/
    │   ├── tile_000.png
    │   ├── tile_001.png
    │   └── ... (up to tile_028.png)
    └── meeples/
        ├── meeple_standing_blue.png
        ├── meeple_lying_blue.png
        └── ... (10 meeple variants)

These unpacked assets serve as the direct prerequisites for `generate_tiles_dataset.py`.
===============================================================================
"""

import os
import shutil
from PIL import Image

def unpack_tile_spritesheet(image_path="tiles_first_edition.webp", output_dir="assets/tiles", cols=23, rows=4, max_tiles=29):
    """
    Slices the Carcassonne 1st edition WebP spritesheet using 23-column grid alignment.
    Purges destination directory before unpacking to ensure clean builds.
    Extracts strictly the first 29 base game tiles (tile_000.png to tile_028.png).
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'")
        return

    # Automatic directory purge prior to extraction
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"[Tile Unpacker] Purged existing directory '{output_dir}/'.")

    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGBA")
    img_w, img_h = img.size

    tile_w = img_w / cols
    tile_h = img_h / rows

    print(f"[Tile Unpacker] Image dimensions: {img_w}x{img_h}px")
    print(f"[Tile Unpacker] Tile pitch: {tile_w:.2f}x{tile_h:.2f}px")

    saved_count = 0

    for r in range(rows):
        for c in range(cols):
            if saved_count >= max_tiles:
                break

            x1 = int(round(c * tile_w))
            y1 = int(round(r * tile_h))
            x2 = int(round((c + 1) * tile_w))
            y2 = int(round((r + 1) * tile_h))

            tile_crop = img.crop((x1, y1, x2, y2))
            tile_resized = tile_crop.resize((64, 64), Image.Resampling.LANCZOS)
            
            filename = os.path.join(output_dir, f"tile_{saved_count:03d}.png")
            tile_resized.save(filename)
            saved_count += 1

        if saved_count >= max_tiles:
            break

    print(f"[Tile Unpacker] Extracted {saved_count} base tile PNGs to '{output_dir}/'.")

def unpack_meeple_spritesheet(image_path="meeples.webp", output_dir="assets/meeples", total_sheet_rows=6, target_rows=2, cols=5):
    """
    Slices the first 2 rows of the meeple WebP spritesheet.
    Purges destination directory before unpacking to ensure clean builds.
    Row 0: Standing Meeples | Row 1: Lying Meeples (Farmers)
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at '{image_path}'")
        return

    # Automatic directory purge prior to extraction
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"[Meeple Unpacker] Purged existing directory '{output_dir}/'.")

    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGBA")
    img_w, img_h = img.size

    sprite_w = img_w / cols
    sprite_h = img_h / total_sheet_rows  # Height derived from full 6-row layout

    colors = ["blue", "green", "black", "red", "yellow"]
    saved_count = 0

    for r in range(target_rows):  # Process rows 0 and 1
        for c in range(cols):
            x1 = int(round(c * sprite_w))
            y1 = int(round(r * sprite_h))
            x2 = int(round((c + 1) * sprite_w))
            y2 = int(round((r + 1) * sprite_h))

            sprite_crop = img.crop((x1, y1, x2, y2))
            color_name = colors[c] if c < len(colors) else f"col_{c}"
            pose = "standing" if r == 0 else "lying"
            
            filename = os.path.join(output_dir, f"meeple_{pose}_{color_name}.png")
            sprite_crop.save(filename)
            saved_count += 1

    print(f"[Meeple Unpacker] Extracted {saved_count} meeple sprite PNGs to '{output_dir}/'.")

if __name__ == "__main__":
    # Unpack first 29 tiles from .webp
    unpack_tile_spritesheet("tiles_first_edition.webp", max_tiles=29)
    
    # Unpack first 2 rows of meeples (standing and lying)
    unpack_meeple_spritesheet("meeples.webp", target_rows=2)