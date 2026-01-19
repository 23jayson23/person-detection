from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import random
import os
import requests

app = FastAPI()

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
    r = requests.get(MODEL_URL, allow_redirects=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded!")

# Load the model
model = YOLO(MODEL_PATH)
PERSON_CLASS_ID = 0

# -----------------------------
# Mock sensor functions
# -----------------------------
def get_temperature():
    return round(random.uniform(20.0, 30.0), 2)

def get_water_level():
    return round(random.uniform(0.0, 100.0), 2)

# -----------------------------
# Person detection endpoint
# -----------------------------
@app.post("/detect")
async def detect_person(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.4)[0]

    persons = []
    for box in results.boxes:
        cls = int(box.cls[0].item())
        if cls != PERSON_CLASS_ID:
            continue  # ignore non-person

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()

        persons.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })

    return {
        "persons": persons,
        "count": len(persons),
        "temperature": get_temperature(),
        "water_level": get_water_level()
    }
