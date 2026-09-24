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
     and class mapping (`class_names.json`) from Hugging Face Model Hub:
     `fcsaba/carcassonne-resnet18-tile-classifier`.

2. Computer Vision Preprocessing & Masking:
   - Uses relative lightness offset (L <= wood_L - 12) to cleanly isolate BGA shaded
     slots from wood background without capturing brown city roofs (CCCCS tile).
   - Applies geometric contour hole filling for solid tile interiors.
   - Calculates Canny boundary edge projections across expanded S pitch (35px-140px)
     and 0.70 candidate threshold to fit fundamental grid pitch.

3. Deep Learning Tile Inference:
   - Slices valid grid cells (> 50% mask coverage), resizes cropped tiles to 64x64px,
     and evaluates them with ResNet18 neural network across 24 tile categories.

4. Browser Extension-Style Visual Grid Export:
   - Displays real-time remaining tile counts in a visual card grid layout with high-contrast
     summary header and "X / Y" concise count formatting (e.g. "3 / 9").

Features:
- Uses @spaces.GPU for ZeroGPU compatibility.
- Auto-triggers analysis upon image file upload / clipboard paste.
- Modern responsive grid UI with opacity dimming for completely played tiles.
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
# HTML & Base64 Visual Card Grid Helpers
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
    Generates a Chrome extension-style responsive visual grid layout for remaining tiles.
    """
    style = """
    <style>
        .carcassonne-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            max-width: 850px;
            margin: 0 auto;
            padding: 10px;
        }
        .summary-bar {
            background-color: #1e293b !important;
            color: #ffffff !important;
            padding: 12px 20px;
            border-radius: 8px;
            text-align: center;
            font-weight: 700 !important;
            font-size: 16px !important;
            margin-bottom: 18px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            letter-spacing: 0.5px;
        }
        .tile-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
            gap: 12px;
        }
        .tile-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .tile-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .tile-card.zero-remaining {
            opacity: 0.45;
            background-color: #f8fafc;
            border-color: #cbd5e1;
        }
        .tile-img {
            width: 54px;
            height: 54px;
            border-radius: 6px;
            object-fit: cover;
            margin-bottom: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        }
        .tile-code {
            font-family: monospace;
            font-weight: 700;
            font-size: 12px;
            color: #334155;
            margin-bottom: 4px;
        }
        .tile-count-info {
            font-size: 14px;
            font-weight: 700;
            color: #2563eb;
        }
        .zero-remaining .tile-count-info {
            color: #64748b;
        }
    </style>
    """

    total_original = sum(ORIGINAL_SET_COMPOSITION.values())
    total_played = sum(played_counts.get(code, 0) for code in ORIGINAL_SET_COMPOSITION)
    total_remaining = total_original - total_played

    html = [
        style,
        '<div class="carcassonne-container">',
        f'<div class="summary-bar">REMAINING TILES: {total_remaining} / {total_original}</div>',
        '<div class="tile-grid">'
    ]

    for tile_code, original_count in ORIGINAL_SET_COMPOSITION.items():
        played_count = played_counts.get(tile_code, 0)
        remaining_count = original_count - played_count

        b64_src = get_tile_image_base64(tile_code)
        img_tag = f'<img src="{b64_src}" class="tile-img" alt="{tile_code}"/>' if b64_src else '<div class="tile-img" style="background:#cbd5e1;"></div>'

        zero_class = " zero-remaining" if remaining_count <= 0 else ""

        html.append(f"""
            <div class="tile-card{zero_class}">
                {img_tag}
                <div class="tile-code">{tile_code}</div>
                <div class="tile-count-info">
                    {remaining_count} / {original_count}
                </div>
            </div>
        """)

    html.append('</div></div>')
    return "".join(html)

# ==============================================================================
# Computer Vision Pipeline Helpers (Masking & Grid Alignment)
# ==============================================================================
def get_real_tile_mask(img):
    """
    Creates a binary mask isolating played tiles from wood and shaded slots.
    Uses relative lightness thresholding (L <= wood_L - 12) to isolate shaded slots
    from wood, while explicitly protecting red roofs and city elements.
    """
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
    is_pure_wood = dist_wood_3d < 16.0
    
    chroma_dist = np.sqrt(
        (img_lab[:, :, 1].astype(np.float32) - wood_lab[1])**2 + 
        (img_lab[:, :, 2].astype(np.float32) - wood_lab[2])**2
    )
    L_channel = img_lab[:, :, 0]
    
    # Shaded placement slots: Same chrominance as wood, but darker (L <= wood_L - 12)
    is_shaded_slot = (chroma_dist < 12.0) & (L_channel <= wood_lab[0] - 12)
    
    is_bg = is_pure_wood | is_shaded_slot
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    is_white_road = (S <= 35) & (V >= 170)
    is_green_field = (H >= 25) & (H <= 90) & (S >= 25) & (V >= 25)
    is_black_meeple = (V < 30)
    is_blue_shield = (H >= 95) & (H <= 130) & (S >= 30) & (V >= 35)
    is_red_roof = ((H <= 15) | (H >= 165)) & (S >= 40) & (V >= 50)
    
    is_tile_pixel = (~is_bg) | is_white_road | is_green_field | is_black_meeple | is_blue_shield | is_red_roof
    raw_mask = (is_tile_pixel.astype(np.uint8)) * 255
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    # Geometric contour hole filling: fills hollow interiors of full-city tiles (CCCCS)
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(clean_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 400:
            cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return filled_mask

def find_optimal_grid_parameters(tile_mask):
    """
    Optimizes 2D grid parameters (Tile Size S, Offset x0, Offset y0)
    by aligning grid lines with the outer boundary edges of played tiles.
    Search pitch range (35px to 140px) handles all screenshot zoom resolutions.
    """
    h, w = tile_mask.shape
    
    mask_edges = cv2.Canny(tile_mask, 100, 200)
    
    v_proj = np.sum(mask_edges, axis=0)
    h_proj = np.sum(mask_edges, axis=1)

    scores_by_S = {}

    for S in range(35, 140):
        best_x_score = -1
        best_x_offset = 0
        for x0 in range(S):
            x_indices = np.arange(x0, w, S)
            if len(x_indices) > 0:
                score_x = np.mean(v_proj[x_indices])
                if score_x > best_x_score:
                    best_x_score = score_x
                    best_x_offset = x0

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

    max_score = max(data[0] for data in scores_by_S.values())

    candidate_S_list = [S for S, (score, _, _) in scores_by_S.items() if score >= 0.70 * max_score]
    best_S = min(candidate_S_list)
    
    _, best_x0, best_y0 = scores_by_S[best_S]

    return best_S, best_x0, best_y0

# ==============================================================================
# Main Board Analysis Engine (Decorated with ZeroGPU runner)
# ==============================================================================
@spaces.GPU
def analyze_carcassonne_board(image_path):
    if image_path is None:
        return "<p style='color:#718096; text-align:center;'>Upload or paste a Carcassonne board screenshot above to automatically view the remaining tile report.</p>"

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

    # Build HTML Visual Card Grid Report
    return build_html_report(played_counts)

# ==============================================================================
# Gradio Web Interface Layout (Auto-triggering Blocks)
# ==============================================================================
with gr.Blocks(title="🏰 Carcassonne BGA Board Analyzer") as demo:
    gr.Markdown("# 🏰 Carcassonne BGA Board Analyzer")
    gr.Markdown("Upload or **paste directly from clipboard (Ctrl+V)** a Board Game Arena (BGA) Carcassonne screenshot to automatically analyze played tiles and calculate the remaining unplayed tile distribution.")
    
    image_input = gr.Image(
        type="filepath", 
        label="Upload or Paste Screenshot (Ctrl+V)",
        sources=["upload", "clipboard"]
    )
    html_output = gr.HTML(label="Analysis Report & Remaining Tile Tally")
    
    # Auto-trigger analysis upon file upload / clipboard paste / change
    image_input.change(
        fn=analyze_carcassonne_board,
        inputs=image_input,
        outputs=html_output
    )

if __name__ == "__main__":
    demo.launch()
