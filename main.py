from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO
from PIL import Image
import base64
import io

app = Flask(__name__)
model = YOLO('yolov8s.pt')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image found'}), 400

    # Decode base64 image
    img_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(img_data)).convert("RGB")

    # Run YOLOv8 detection
    results = model(image)

    # Ambil label deteksi dan confidence
    labels = results[0].names
    detections = []
    for r in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        detections.append({
            'label': labels[int(class_id)],
            'confidence': round(score, 2)
        })

    return jsonify({'detections': detections})
