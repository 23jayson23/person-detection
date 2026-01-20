from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import random
import os
import requests

app = FastAPI(title="YOLO Chick Detection API")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Download YOLO model dynamically
# -----------------------------
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODEL_PATH = "yolov8n.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model...")
    r = requests.get(MODEL_URL, timeout=30)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded!")

# -----------------------------
# Load YOLO once
# -----------------------------
model = YOLO(MODEL_PATH)
CHICKEN_CLASS_ID = 14  # COCO class ID for chicken

# -----------------------------
# Mock sensor functions
# -----------------------------
def get_temperature():
    return round(random.uniform(20.0, 30.0), 2)

def get_water_level():
    return round(random.uniform(0.0, 100.0), 2)

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}

# -----------------------------
# Chick detection endpoint
# -----------------------------
@app.post("/detect")
async def detect_chick(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    results = model(frame, conf=0.4)[0]

    chicks = []
    for box in results.boxes:
        if int(box.cls[0]) != CHICKEN_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1

        # Filter by size: consider only small chickens as chicks
        if width < 50 and height < 50:  # Adjust these thresholds as needed
            confidence = float(box.conf[0])
            chicks.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            })

    return {
        "chicks": chicks,
        "count": len(chicks),
        "temperature": get_temperature(),
        "water_level": get_water_level()
    }
