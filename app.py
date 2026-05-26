import os
import time
import warnings
import traceback

# ================= ENV (REMOVE TENSORFLOW WARNINGS) =================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings('ignore')

# ================= IMPORTS =================
from flask import Flask, render_template, request, jsonify, url_for
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
import json
from werkzeug.utils import secure_filename

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= FLASK APP =================
app = Flask(__name__)

# ================= UPLOAD FOLDER =================
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= FOOD DATA =================
food_data = {}

# ================= SAFE MODEL LOADING =================
print("Loading models...")

try:
    custom_cnn_model = tf.keras.models.load_model('models/custom_weights.keras')
    resnet_model = tf.keras.models.load_model('models/resnet_weights.keras')
    vgg16_model = tf.keras.models.load_model('models/vgg16_weights.keras')
    print("✅ Models Loaded Successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    custom_cnn_model = None
    resnet_model = None
    vgg16_model = None

# ================= CLASS MAPPING =================
with open('json/class_indices.json', 'r') as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}
class_names_list = list(class_names.values())

# ================= LOAD METRICS =================
with open('json/custom_CNN_metrics.json', 'r') as f:
    custom_metrics = json.load(f)

with open('json/resnet_CNN_metrics.json', 'r') as f:
    resnet_metrics = json.load(f)

with open('json/vgg16_CNN_metrics.json', 'r') as f:
    vgg16_metrics = json.load(f)

# ================= HELPERS =================
def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def extract_class_metrics(metrics_data, predicted_class):
    block = metrics_data.get(predicted_class, {})

    if not block:
        for key in metrics_data:
            if key.lower() == predicted_class.lower():
                block = metrics_data[key]
                break

    if block and "classification_report" in block:
        report = block["classification_report"]
        return {
            "precision": round(float(report.get("precision", 0)), 2),
            "recall": round(float(report.get("recall", 0)), 2),
            "f1-score": round(float(report.get("f1-score", 0)), 2)
        }

    return {"precision": "N/A", "recall": "N/A", "f1-score": "N/A"}


def extract_accuracy(metrics_data, predicted_class):
    block = metrics_data.get(predicted_class, {})

    if not block:
        for key in metrics_data:
            if key.lower() == predicted_class.lower():
                block = metrics_data[key]
                break

    if block and "overall_model_accuracy" in block:
        return round(float(block["overall_model_accuracy"]) * 100, 2)

    return "N/A"

# ================= HOME ROUTE =================
@app.route('/')
def home():
    return render_template('index.html', class_names=class_names_list)

# ================= PREDICT ROUTE =================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"})

        image = request.files['image']
        model_name = request.form.get('model')

        # 🔥 UNIQUE FILE NAME (PREVENT CACHE ISSUE)
        filename = str(int(time.time())) + "_" + secure_filename(image.filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        img = preprocess_image(filepath)

        # ================= MODEL SELECTION =================
        if model_name == 'customcnn':
            model = custom_cnn_model
            metrics_data = custom_metrics
        elif model_name == 'resnet':
            model = resnet_model
            metrics_data = resnet_metrics
        elif model_name == 'vgg16':
            model = vgg16_model
            metrics_data = vgg16_metrics
        else:
            return jsonify({"error": "Invalid model selected"})

        # ================= SAFETY CHECK =================
        if model is None:
            return jsonify({"error": "Model not loaded properly"})

        # ================= PREDICTION =================
        pred = model.predict(img)

        index = int(np.argmax(pred))
        confidence = round(float(np.max(pred)) * 100, 2)

        predicted_class = class_names.get(index, "Unknown")

        # 🔥 CORRECT IMAGE PATH (IMPORTANT FIX)
        detected_image = url_for('static', filename='uploads/' + filename)

        class_report = extract_class_metrics(metrics_data, predicted_class)
        accuracy = extract_accuracy(metrics_data, predicted_class)

        nutrition = food_data.get(predicted_class, {
            "Calories": "N/A",
            "Protein": "N/A",
            "Fat": "N/A"
        })

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "accuracy": accuracy,
            "detected_image": detected_image,
            "nutrition": nutrition,
            "class_report": class_report
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

# ================= RUN APP (RENDER SAFE) =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)