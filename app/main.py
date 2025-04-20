'''
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO  # or appropriate loader if you're using YOLOv11 directly
import cv2


app = Flask(__name__)
CORS(app)  # Enable CORS to allow the frontend to make requests

model = YOLO("./best.pt")

try:
    interpreter = tf.lite.Interpreter(model)
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Model loading failed: {str(e)}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess the image using EfficientNet's preprocessing
    image = image.convert('RGB')  # Ensure the image is in RGB format
    image = image.resize((800, 800))  # Resize to model's input size
    image_array = np.array(image)
    image_array = preprocess_input(image_array)  # Apply EfficientNet preprocessing
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Debugging: Print image array details
    print(f"Image Array Shape: {image_array.shape}, Min: {np.min(image_array)}, Max: {np.max(image_array)}")

# Run YOLO inference
    results = model(image)[0]

    # Process results
    predictions = []
    for box in results.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        
        predictions.append({
            "class": model.names[class_id],
            "confidence": round(confidence, 2),
            "bbox": [int(b) for b in bbox]
        })

    return JSONResponse(content=predictions)
'''
'''
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load YOLO model
yolo_model = YOLO("./best.pt")

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))

    # Preprocess image for YOLO
    image = image.convert('RGB')
    image = image.resize((800, 800))  # Resize to YOLO input size if needed
    image_array = np.array(image)

    # Debugging info
    print(f"Image shape: {image_array.shape}, dtype: {image_array.dtype}")

    # Run YOLO prediction
    results = yolo_model(image)[0]

    predictions = []
    for box in results.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        predictions.append({
            "class": yolo_model.names[class_id],
            "confidence": round(confidence, 2),
            "bbox": [int(b) for b in bbox]
        })

    return jsonify(predictions)
'''

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load YOLO model once at startup
yolo_model = YOLO("./best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('RGB')
        image = image.resize((800, 800))  # Optional resize if model expects specific input size
    except Exception as e:
        return jsonify({'error': f'Invalid image format. {str(e)}'}), 400

    # Run YOLO prediction
    try:
        results = yolo_model(image)[0]
    except Exception as e:
        return jsonify({'error': f'YOLO prediction failed. {str(e)}'}), 500

    # Extract predictions
    predictions = []
    for box in results.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        predictions.append({
            "class": yolo_model.names.get(class_id, str(class_id)),
            "confidence": round(confidence, 2),
            "bbox": [int(b) for b in bbox]
        })

    return jsonify(predictions)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
