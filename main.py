from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")

PERSON_CLASS_ID = 0

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
            continue  # ❌ ignore non-person

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()

        persons.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": conf
        })

    return {
        "persons": persons,
        "count": len(persons)
    }
