import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFile
import json
import warnings
from werkzeug.utils import secure_filename
import traceback

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading models...")

custom_cnn_model = tf.keras.models.load_model('models/custom_weights.keras', compile=False)
resnet_model     = tf.keras.models.load_model('models/resnet_weights.keras', compile=False)
vgg16_model      = tf.keras.models.load_model('models/vgg16_weights.keras', compile=False)

print("✅ Models Loaded")

# ================= JSON SAFE LOAD =================
def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        print(f"❌ Missing file: {path}")
        return {}

class_indices = load_json('json/class_indices.json')
custom_metrics = load_json('json/custom_CNN_metrics.json')
resnet_metrics = load_json('json/resnet_CNN_metrics.json')
vgg16_metrics = load_json('json/vgg16_CNN_metrics.json')

class_names = {v: k for k, v in class_indices.items()}
class_names_list = list(class_names.values())

# ================= IMAGE =================
def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ================= ROUTES =================
@app.route('/')
def home():
    return render_template('index.html', class_names=class_names_list)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image']
        model_name = request.form.get('model')

        filename = secure_filename(image.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        img = preprocess_image(filepath)

        if model_name == 'customcnn':
            model = custom_cnn_model
        elif model_name == 'resnet':
            model = resnet_model
        else:
            model = vgg16_model

        pred = model.predict(img)

        index = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100

        predicted_class = class_names.get(index, "Unknown")

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "detected_image": "/" + filepath
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)