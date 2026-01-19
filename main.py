from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")

CHICKEN_CLASS_ID = 14  # bird in COCO

# Thresholds
MAX_TEMP = 32.0
MIN_LEVEL = 30  # percent
MAX_CHICK_AREA_RATIO = 0.15

def get_environment_and_alerts():
    temperature = round(random.uniform(26.0, 36.0), 1)
    feed_level = random.randint(10, 100)
    water_level = random.randint(10, 100)

    alerts = []

    # Temperature alert
    if temperature > MAX_TEMP:
        alerts.append({
            "type": "TEMPERATURE_HIGH",
            "message": f"High temperature detected: {temperature}°C",
            "severity": "CRITICAL"
        })

    # Feed alert
    if feed_level <= MIN_LEVEL:
        alerts.append({
            "type": "FEED_LOW",
            "message": f"Feed level is low: {feed_level}%",
            "severity": "WARNING"
        })

    # Water alert
    if water_level <= MIN_LEVEL:
        alerts.append({
            "type": "WATER_LOW",
            "message": f"Water level is low: {water_level}%",
            "severity": "WARNING"
        })

    environment = {
        "temperature": {
            "value": temperature,
            "unit": "°C",
            "status": "HIGH" if temperature > MAX_TEMP else "NORMAL"
        },
        "feeding": {
            "level_percent": feed_level,
            "status": "LOW" if feed_level <= MIN_LEVEL else "OK"
        },
        "drinking_water": {
            "level_percent": water_level,
            "status": "LOW" if water_level <= MIN_LEVEL else "OK"
        }
    }

    return environment, alerts


@app.post("/detect")
async def detect_chicken(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.4)[0]

    chickens = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != CHICKEN_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])

        chickens.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": round(conf, 3)
        })

    environment, alerts = get_environment_and_alerts()

    return {
        "chickens": chickens,
        "count": len(chickens),
        "environment": environment,
        "alerts": alerts,
        "has_alerts": len(alerts) > 0
    }
