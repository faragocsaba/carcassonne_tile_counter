"""
===============================================================================
Carcassonne BGA Board Analyzer - Hugging Face Gradio Space (app.py)
===============================================================================

Dependencies & Requirements (requirements.txt):
----------------------------------------------
torch
torchvision
pillow
opencv-python-headless
numpy
huggingface_hub
gradio

Inputs:
-------
- Full Board Game Arena (BGA) Carcassonne match screenshot (.jpg / .jpeg)

Workflow & Processing Architecture:
-----------------------------------
1. Pretrained Model & Mapping Retrieval:
   - Automatically downloads fine-tuned ResNet18 model weights (`carcassonne_model.pth`)
     and class mapping (`class_names.json`) from the Hugging Face Model Hub repository:
     `fcsaba/carcassonne-resnet18-tile-classifier`.

2. Computer Vision Preprocessing:
   - Performs LAB chrominance color masking (`get_real_tile_mask`) to isolate played
     tiles from both light wood backgrounds and BGA darkened placement slots.
   - Calculates Canny boundary edge projections to optimize 2D grid pitch S (80-140px)
     and phase origin offsets (x0, y0) using `find_optimal_grid_parameters`.

3. Deep Learning Tile Inference:
   - Slices valid grid cells (> 50% mask coverage), resizes cropped tiles to 64x64px,
     and evaluates them with the ResNet18 neural network across 24 tile categories.

4. Set Composition & Remaining Tile Analysis:
   - Tabulates played tile counts per class and subtracts them from the official
     72-tile base game distribution to produce a real-time remaining tile report.

Outputs:
--------
- Annotated image overlay displaying green bounding boxes around recognized tiles.
- Formatted ASCII text table listing played, canonical total, and remaining counts.

Features:
- Uses @spaces.GPU for ZeroGPU compatibility.
- Auto-triggers analysis upon image file upload/change.
- Displays HTML report table with embedded Base64 tile preview icons.
- Improved high-contrast table header styling with CSS !important rules.
===============================================================================
"""

import spaces
import os
import json
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from huggingface_hub import hf_hub_download
import gradio as gr
from collections import Counter

# ==============================================================================
# Global Configuration & Model Download from Hugging Face Hub
# ==============================================================================
HF_REPO_ID = "fcsaba/carcassonne-resnet18-tile-classifier"
IMAGE_SIZE = (64, 64)
TILES_DIR = "tiles"

# Official Carcassonne 1st Edition Base Set distribution (Total = 72 land tiles)
ORIGINAL_SET_COMPOSITION = {
    "CCCCS": 1, "RRRR": 1, "CCCF": 3, "CCCFS": 1,  "CCCR": 1, "CCCRS": 2,
    "RRRF": 4, "CFCF": 1, "CFCFS": 2, "RFRF": 8, "CCFF": 3, "CCFFS": 2,
    "CCRR": 3, "CCRRS": 2, "RRFF": 9, "CCFF2": 2, "CFCF2": 3, "RFFF": 2,
    "FFFF": 4, "CFFF": 5, "CRRF": 3, "CFRR": 3, "CRRR": 3, "CRFR": 4,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download model artifacts from Hugging Face Hub
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="carcassonne_model.pth")
classes_path = hf_hub_download(repo_id=HF_REPO_ID, filename="class_names.json")

with open(classes_path, "r") as f:
    idx_to_class = json.load(f)

# Reconstruct ResNet18 architecture for inference
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, len(idx_to_class))
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# PyTorch Image Preprocessing Transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# HTML & Base64 Helpers
# ==============================================================================
def get_tile_image_base64(tile_code):
    """
    Reads PNG reference image from `tiles/<tile_code>.png` and converts it to
    a Base64 data URI string for inline HTML rendering.
    """
    img_path = os.path.join(TILES_DIR, f"{tile_code}.png")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            encoded_str = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded_str}"
    return ""

def build_html_report(played_counts):
    """
    Generates a stylized HTML table with embedded tile icons and high-contrast header styling.
    """
    style = """
    <style>
        .carcassonne-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 700px;
            margin: 0 auto;
            padding-top: 10px;
        }
        .carcassonne-table {
            width: 100%;
            border-collapse: collapse;
            background-color: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .carcassonne-table th {
            background-color: #1e293b !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            text-align: center !important;
            padding: 12px 16px !important;
            font-size: 15px !important;
            letter-spacing: 0.5px !important;
        }
        .carcassonne-table td {
            padding: 8px 16px;
            border-bottom: 1px solid #e2e8f0;
            vertical-align: middle;
            text-align: center;
            font-size: 14px;
            color: #2d3748;
        }
        .carcassonne-table tr:nth-child(even) {
            background-color: #f8fafc;
        }
        .carcassonne-table tr:hover {
            background-color: #edf2f7;
        }
        .tile-icon {
            width: 44px;
            height: 44px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
            object-fit: cover;
            vertical-align: middle;
        }
        .tile-code {
            font-weight: 700;
            font-family: monospace;
            color: #1a202c;
            font-size: 15px;
        }
        .val-played {
            font-weight: 600;
            color: #c53030;
        }
        .val-remaining {
            font-weight: 700;
            color: #2b6cb0;
        }
        .total-row td {
            font-weight: bold;
            background-color: #e2e8f0;
            border-top: 2px solid #cbd5e0;
            font-size: 15px;
            color: #1a202c;
        }
    </style>
    """

    html = [style, '<div class="carcassonne-container">', '<table class="carcassonne-table">']
    html.append("""
        <thead>
            <tr>
                <th>Tile Icon</th>
                <th>Type Code</th>
                <th>Played</th>
                <th>Set Total</th>
                <th>Remaining</th>
            </tr>
        </thead>
        <tbody>
    """)

    total_original = sum(ORIGINAL_SET_COMPOSITION.values())
    total_played = 0
    total_remaining = total_original

    for tile_code, original_count in ORIGINAL_SET_COMPOSITION.items():
        played_count = played_counts.get(tile_code, 0)
        remaining_count = original_count - played_count

        total_played += played_count
        total_remaining -= played_count

        b64_src = get_tile_image_base64(tile_code)
        img_tag = f'<img src="{b64_src}" class="tile-icon" alt="{tile_code}"/>' if b64_src else '-'

        html.append(f"""
            <tr>
                <td>{img_tag}</td>
                <td class="tile-code">{tile_code}</td>
                <td class="val-played">{played_count}</td>
                <td>{original_count}</td>
                <td class="val-remaining">{remaining_count}</td>
            </tr>
        """)

    html.append(f"""
            <tr class="total-row">
                <td colspan="2">TOTAL LAND TILES</td>
                <td class="val-played">{total_played}</td>
                <td>{total_original}</td>
                <td class="val-remaining">{total_remaining}</td>
            </tr>
        </tbody>
    </table>
    </div>
    """)

    return "".join(html)

# ==============================================================================
# Computer Vision Pipeline Helpers (Masking & Grid Alignment)
# ==============================================================================
def get_real_tile_mask(img):
    h, w, _ = img.shape
    corner_size = 30
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
    h, w = tile_mask.shape
    mask_edges = cv2.Canny(tile_mask, 100, 200)
    v_proj = np.sum(mask_edges, axis=0)
    h_proj = np.sum(mask_edges, axis=1)
    
    best_overall_score = -1
    best_S, best_x0, best_y0 = 100, 0, 0
    
    for S in range(80, 140):
        best_x_score, best_x_offset = -1, 0
        for x0 in range(S):
            score_x = np.sum(v_proj[np.arange(x0, w, S)])
            if score_x > best_x_score: 
                best_x_score = score_x
                best_x_offset = x0
            
        best_y_score, best_y_offset = -1, 0
        for y0 in range(S):
            score_y = np.sum(h_proj[np.arange(y0, h, S)])
            if score_y > best_y_score: 
                best_y_score = score_y
                best_y_offset = y0
            
        total_score = best_x_score + best_y_score
        if total_score > best_overall_score:
            best_overall_score = total_score
            best_S, best_x0, best_y0 = S, best_x_offset, best_y_offset
            
    return best_S, best_x0, best_y0

# ==============================================================================
# Main Board Analysis Engine (Decorated with ZeroGPU runner)
# ==============================================================================
@spaces.GPU
def analyze_carcassonne_board(image_path):
    if image_path is None:
        return "<p style='color:#718096; text-align:center;'>Upload a Carcassonne board screenshot above to automatically view the remaining tile report.</p>"

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        return "<p style='color:red; text-align:center;'>Error: Unable to load the uploaded image file.</p>"

    height, width, _ = img_cv2.shape
    
    tile_mask = get_real_tile_mask(img_cv2)
    tile_pixels = np.column_stack(np.where(tile_mask > 0))
    
    if len(tile_pixels) == 0:
        return "<p style='color:red; text-align:center;'>No valid Carcassonne tiles detected on the board image.</p>"

    min_y, min_x = tile_pixels.min(axis=0)
    max_y, max_x = tile_pixels.max(axis=0)
    
    tile_size, x0, y0 = find_optimal_grid_parameters(tile_mask)
    
    col_start = int(np.floor((min_x - x0) / tile_size))
    col_end = int(np.ceil((max_x - x0) / tile_size))
    row_start = int(np.floor((min_y - y0) / tile_size))
    row_end = int(np.ceil((max_y - y0) / tile_size))
    
    pil_img_src = Image.open(image_path)
    played_counts = Counter()
    
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
            if fill_ratio > 0.50:
                crop_box = (x1, y1, x2, y2)
                tile_crop = pil_img_src.crop(crop_box).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                # Model Inference
                img_rgb = tile_crop.convert("RGB")
                input_tensor = transform(img_rgb).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    pred_idx = torch.argmax(outputs, dim=1).item()
                
                label = idx_to_class[str(pred_idx)]
                played_counts[label] += 1

    # Build HTML Table Report
    return build_html_report(played_counts)

# ==============================================================================
# Gradio Web Interface Layout (Auto-triggering Blocks)
# ==============================================================================
with gr.Blocks(title="🏰 Carcassonne BGA Board Analyzer") as demo:
    gr.Markdown("# 🏰 Carcassonne BGA Board Analyzer")
    gr.Markdown("Upload a Board Game Arena (BGA) Carcassonne screenshot to automatically analyze played tiles and calculate the remaining unplayed tile distribution.")
    
    image_input = gr.Image(type="filepath", label="Upload BGA Board Screenshot")
    html_output = gr.HTML(label="Analysis Report & Remaining Tile Tally")
    
    # Auto-trigger analysis upon file upload/change
    image_input.change(
        fn=analyze_carcassonne_board,
        inputs=image_input,
        outputs=html_output
    )

if __name__ == "__main__":
    demo.launch()