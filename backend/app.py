"""
Handwritten-to-Digital Converter - Flask Backend
Receives a canvas/image, segments it into individual characters using
OpenCV contour detection, classifies each character with the trained
CNN, reconstructs the string, and (if it parses as a math expression)
evaluates it safely.
"""

import io
import json
import re
import base64
import hashlib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------
# Model definition (must match training script exactly)
# ---------------------------------------------------------------
class CharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------------
# Load model + class map at startup
# ---------------------------------------------------------------
with open("model/class_map.json") as f:
    RAW_CLASS_MAP = json.load(f)
CLASS_MAP = {int(k): v for k, v in RAW_CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)

model = CharCNN(NUM_CLASSES)
model.load_state_dict(torch.load("model/char_cnn.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"Loaded model with {NUM_CLASSES} classes")

# Only these characters are allowed through to eval() -- everything else
# is treated as plain recognized text, never executed.
SAFE_MATH_PATTERN = re.compile(r"^[0-9+\-*/().\s]+$")


# ---------------------------------------------------------------
# Segmentation: split a full canvas image into per-character crops
# ---------------------------------------------------------------
def segment_characters(image_np):
    """
    image_np: grayscale numpy array, ink = white (255), background = black (0)
    Returns list of (x, cropped_32x32_array) sorted left-to-right,
    so downstream code can reconstruct reading order.
    """
    _, binary = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY)
    print(f"DEBUG binary: min={binary.min()}, max={binary.max()}, nonzero={np.count_nonzero(binary)}, total={binary.size}")
    cv2.imwrite("debug_binary_full.png", binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"DEBUG contours found: {len(contours)}")
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"  contour {i}: x={x}, y={y}, w={w}, h={h}")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 4 or h < 4:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[0])

    crops = []
    for (x, y, w, h) in boxes:
        char_img = binary[y:y + h, x:x + w]

        margin = int(max(w, h) * 0.25)
        size = max(w, h) + 2 * margin
        square = np.zeros((size, size), dtype=np.uint8)
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        square[y_off:y_off + h, x_off:x_off + w] = char_img

        resized = cv2.resize(square, (32, 32), interpolation=cv2.INTER_AREA)
        crops.append((x, resized))

    return crops

def predict_char(char_img_np):
    """char_img_np: 32x32 grayscale numpy array, ink=white. Returns (char, confidence)."""
    tensor = char_img_np.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,32,32]

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    label = CLASS_MAP.get(pred_idx.item(), "?")
    return label, conf.item()


def try_evaluate_expression(text):
    """Safely evaluate only if the recognized text is a pure arithmetic expression."""
    cleaned = text.replace("x", "*").replace("X", "*")  # common OCR confusion for times
    if not SAFE_MATH_PATTERN.match(cleaned):
        return None
    try:
        # eval is safe here ONLY because SAFE_MATH_PATTERN whitelists digits,
        # +-*/(). and whitespace -- no names, no attribute access, no builtins.
        result = eval(cleaned, {"__builtins__": {}}, {})
        return result
    except Exception:
        return None


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "num_classes": NUM_CLASSES})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "image": "data:image/png;base64,...." }
    Canvas image should have dark background, light/white strokes
    (or we invert automatically if it looks like the opposite).
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    img_b64 = data["image"].split(",")[-1]
    img_bytes = base64.b64decode(img_b64)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale
    img_np = np.array(pil_img)

    cv2.imwrite("debug_raw_decoded.png", img_np)
    print(f"DEBUG raw decoded: min={img_np.min()}, max={img_np.max()}, mean={img_np.mean():.1f}")

    import hashlib
    img_hash = hashlib.md5(img_np.tobytes()).hexdigest()[:8]
    print(f"Received image hash: {img_hash}, shape: {img_np.shape}, mean: {img_np.mean():.1f}, nonzero_pixels: {np.count_nonzero(img_np)}")
   
    # Auto-detect polarity: if background is mostly bright, invert
    # so ink is always white-on-black for consistency with training data.
    if img_np.mean() > 127:
        img_np = 255 - img_np

    crops = segment_characters(img_np)
    if not crops:
        return jsonify({"text": "", "expression_result": None, "characters": []})

        # DEBUG: save the raw crop as an actual image file we can visually inspect
    for i, (x, crop) in enumerate(crops):
        cv2.imwrite(f"debug_crop_{i}_latest.png", crop)
        print(f"  Saved debug_crop_{i}_latest.png")

    print(f"\n=== NEW PREDICTION REQUEST ===")
    print(f"Number of segmented characters detected: {len(crops)}")

    print(f"\n=== NEW PREDICTION REQUEST ===")
    print(f"Number of segmented characters detected: {len(crops)}")

    results = []
    reconstructed = ""
    for i, (x, crop) in enumerate(crops):
        crop_hash = hashlib.md5(crop.tobytes()).hexdigest()[:8]
        char, conf = predict_char(crop)
        results.append({"char": char, "confidence": round(conf, 4)})
        reconstructed += char
        print(f"  Crop {i}: hash={crop_hash}, nonzero={np.count_nonzero(crop)}, predicted={char!r}, confidence={conf:.4f}")

    print(f"Final reconstructed text: {reconstructed!r}\n")

    expr_result = try_evaluate_expression(reconstructed)

    return jsonify({
        "text": reconstructed,
        "characters": results,
        "expression_result": expr_result,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
