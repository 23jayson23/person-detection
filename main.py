from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import random
import os
import requests
from inference_sdk import InferenceHTTPClient
import tempfile

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(title="Chick Detection API (Roboflow + YOLO)")

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
# Roboflow Inference Client
# -----------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="qlZgeVhDIk2uOfski0Y2"
)

WORKSPACE_NAME = "personal-sld3z"
WORKFLOW_ID = "find-chicks"

# -----------------------------
# YOLO Setup (fallback)
# -----------------------------
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
MODEL_PATH = "yolov8n.pt"
CHICKEN_CLASS_ID = 14  # COCO class ID for chicken

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model...")
    r = requests.get(MODEL_URL, timeout=30)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded!")

yolo_model = YOLO(MODEL_PATH)

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
    # Save uploaded image temporarily
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # -------------------------
        # Try Roboflow first
        # -------------------------
        try:
            rf_result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={"image": tmp_path},
                use_cache=True
            )

            chicks = []
            for detection in rf_result.get("predictions", []):
                if detection.get("class") == "chick":  # adjust based on your Roboflow workflow class
                    chicks.append({
                        "bbox": detection.get("bbox"),
                        "confidence": detection.get("confidence")
                    })

            return {
                "source": "roboflow",
                "chicks": chicks,
                "count": len(chicks),
                "temperature": get_temperature(),
                "water_level": get_water_level()
            }

        except Exception as e:
            print(f"Roboflow failed: {e}. Falling back to YOLO.")

        # -------------------------
        # Fallback: YOLO detection
        # -------------------------
        frame = cv2.imread(tmp_path)
        if frame is None:
            return {"error": "Invalid image"}

        results = yolo_model(frame, conf=0.4)[0]

        chicks = []
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            if cls_id != CHICKEN_CLASS_ID:
                continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            width = x2 - x1
            height = y2 - y1

            # Filter by size: consider only small chickens as chicks
            if width < 50 and height < 50:  # adjust thresholds as needed
                confidence = float(box.conf.cpu().numpy()[0])
                chicks.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence
                })

        return {
            "source": "yolo",
            "chicks": chicks,
            "count": len(chicks),
            "temperature": get_temperature(),
            "water_level": get_water_level()
        }

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
