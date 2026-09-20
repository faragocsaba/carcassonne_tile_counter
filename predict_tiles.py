"""
===============================================================================
Carcassonne Tile Predictor - ResNet18 Inference (predict_tiles.py)
===============================================================================
"""

import os
import glob
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

MODEL_PATH = "carcassonne_model.pth"
CLASSES_PATH = "class_names.json"
INPUT_DIR = "extracted_tiles"
OUTPUT_DIR = "classified_results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Class Names
with open(CLASSES_PATH, "r") as f:
    idx_to_class = json.load(f)

num_classes = len(idx_to_class)

# Define Pretrained ResNet18 Architecture
def get_resnet_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

model = get_resnet_model(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_paths = sorted(glob.glob(f"{INPUT_DIR}/*/*.png"))
image_paths = [p for p in image_paths if not os.path.basename(p).startswith("debug_")]

print(f"[Inference] Classifying {len(image_paths)} extracted tiles using ResNet18...")

for img_path in image_paths:
    pil_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probabilities, 1)

    predicted_label = idx_to_class[str(pred_idx.item())]
    confidence_score = conf.item() * 100

    rel_path = os.path.relpath(img_path, INPUT_DIR)
    print(f"File: {rel_path:35s} -> Predicted: {predicted_label:8s} ({confidence_score:.1f}%)")

    save_dir = os.path.join(OUTPUT_DIR, predicted_label)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{confidence_score:.0f}pct_{os.path.basename(img_path)}")
    pil_img.save(out_path)

print(f"\n[Inference] 🎉 Classification finished! Results stored in '{OUTPUT_DIR}/'.")
