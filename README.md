
<h1 align="center">🍔 Food Classification using Deep Learning 🍕</h1>

<p align="center">
Deep Learning based Food Image Classification System using CNN, VGG16 and ResNet50
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
This project is a <b>Deep Learning-based Food Classification System</b>
developed using <b>TensorFlow, Keras, CNN, VGG16, and ResNet50</b>.
The system classifies food images into multiple food categories and
generates nutritional information such as calories, protein, fat,
carbohydrates, and fiber values.
</p>

<ul>
<li>Custom CNN Model</li>
<li>VGG16 Transfer Learning</li>
<li>ResNet50 Transfer Learning</li>
<li>Food Nutrition Generator</li>
<li>Image Prediction System</li>
</ul>

<hr>

<h2>🚀 Features</h2>

<ul>
<li>✅ Food Image Classification</li>
<li>✅ 34 Food Categories</li>
<li>✅ CNN Architecture</li>
<li>✅ VGG16 Transfer Learning</li>
<li>✅ ResNet50 Transfer Learning</li>
<li>✅ Nutritional Information Generator</li>
<li>✅ Data Augmentation</li>
<li>✅ Model Evaluation Metrics</li>
<li>✅ Image Prediction System</li>
</ul>

<hr>

<h2>🧠 Technologies Used</h2>

<ul>
<li>Python</li>
<li>TensorFlow</li>
<li>Keras</li>
<li>OpenCV</li>
<li>NumPy</li>
<li>Matplotlib</li>
<li>Scikit-learn</li>
<li>Google Colab</li>
</ul>

<hr>

<h2>📂 Dataset Structure</h2>

<pre>
New_dataset/
│
├── train/
│   ├── pizza/
│   ├── burger/
│   ├── momos/
│   └── ...
│
├── val/
│   ├── pizza/
│   ├── burger/
│   ├── momos/
│   └── ...
</pre>

<hr>

<h2>🍽️ Food Categories</h2>

<p>
Apple Pie, Baked Potato, Burger, Butter Naan, Chai,
Chapati, Cheesecake, Chicken Curry, Chole Bhature,
Crispy Chicken, Dal Makhani, Dhokla, Donut,
Fried Rice, Fries, Hot Dog, Ice Cream, Idli,
Jalebi, Kaathi Rolls, Kadai Paneer, Kulfi,
Masala Dosa, Momos, Omelette, Paani Puri,
Pakode, Pav Bhaji, Pizza, Samosa,
Sandwich, Sushi, Taco and Taquito.
</p>

<hr>

<h2>🔄 Data Preprocessing</h2>

<ul>
<li>Image Resizing</li>
<li>Image Normalization</li>
<li>Data Augmentation</li>
<li>Train-Validation Split</li>
<li>Dataset Cleaning</li>
</ul>

<h3>Data Augmentation</h3>

<pre>
ImageDataGenerator(
    rescale=1/255,
    rotation_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)
</pre>

<hr>

<h2>🏗️ Models Used</h2>

<h3>1️⃣ Custom CNN Model</h3>

<ul>
<li>Conv2D Layers</li>
<li>MaxPooling Layers</li>
<li>Flatten Layer</li>
<li>Dense Layers</li>
<li>Softmax Output Layer</li>
</ul>

<h3>2️⃣ VGG16 Transfer Learning</h3>

<ul>
<li>Pretrained on ImageNet</li>
<li>Better Feature Extraction</li>
<li>Improved Classification Accuracy</li>
</ul>

<h3>3️⃣ ResNet50 Transfer Learning</h3>

<ul>
<li>Deep Residual Architecture</li>
<li>Enhanced Learning Capability</li>
<li>Improved Validation Performance</li>
</ul>

<hr>

<h2>📊 Model Evaluation</h2>

<pre>
accuracy_score()
confusion_matrix()
classification_report()
</pre>

<ul>
<li>Accuracy</li>
<li>Precision</li>
<li>Recall</li>
<li>F1-Score</li>
<li>Support</li>
</ul>

<hr>

<h2>🥗 Nutritional Information Generator</h2>

<pre>
{
    "pizza": {
        "calories": 320,
        "protein": 12.5,
        "fat": 14.2,
        "carbs": 36.8,
        "fiber": 3.5
    }
}
</pre>

<hr>

<h2>🔍 Prediction System</h2>

<h3>Prediction Function</h3>

<pre>
pred_image("/content/momos.jpeg")
</pre>

<h3>Output</h3>

<pre>
momos
</pre>

<hr>

<h2>💾 Saved Models</h2>

<pre>
vGG16.ipynb_weights.keras
resnet_weights.keras
custom_weights.keras
</pre>

<hr>

<h2>📈 Future Improvements</h2>

<ul>
<li>Real-time Camera Prediction</li>
<li>Flask/Streamlit Deployment</li>
<li>EfficientNet Integration</li>
<li>Mobile Application</li>
<li>Calorie Recommendation System</li>
</ul>

<hr>

<h2>▶️ How to Run</h2>

<h3>1️⃣ Install Dependencies</h3>

<pre>
pip install tensorflow keras opencv-python matplotlib scikit-learn
</pre>

<h3>2️⃣ Run Training</h3>

<pre>
python train.py
</pre>

<h3>3️⃣ Run Prediction</h3>

<pre>
python predict.py
</pre>

<hr>

<h2>📸 Sample Prediction</h2>

<pre>
pred_image("/content/momos.jpeg")
</pre>

<p><b>Predicted Output:</b></p>

<pre>
Momos
</pre>

<hr>

<h2>👩‍💻 Author</h2>

<p>
<b>Shaik Rizwana</b><br>
Computer Science Graduate<br>
Machine Learning Enthusiast<br>
Python Developer
</p>

<hr>

<h2>⭐ Conclusion</h2>

<p>
This project demonstrates the implementation of Deep Learning techniques
for Food Image Classification using CNN and Transfer Learning models like
VGG16 and ResNet50. The system accurately classifies food images and
generates nutritional information for smart food recognition applications.
</p>

<hr>

