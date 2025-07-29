# 🌿 Plant Disease Detection using Machine Learning

## 📌 Overview

This project focuses on building a **Machine Learning-based system** to detect plant diseases from leaf images. The goal is to help farmers and agricultural experts quickly identify diseases and take timely actions to protect crops.

The project uses image processing techniques for feature extraction and classification algorithms to predict the type of disease affecting a plant.

---

## 🔍 Features

- 📷 Upload leaf image for disease prediction  
- 🧠 Machine Learning-based classification  
- 🌱 Identifies common diseases in plants like tomato, potato, and corn  
- 📊 Provides probability scores for disease classes  
- 💡 Easy-to-use and lightweight system for practical deployment

---

## 🧰 Tech Stack

- **Programming Language**: Python  
- **Libraries Used**:
  - `scikit-learn`
  - `numpy`
  - `matplotlib`
  - `pandas`
  - `OpenCV` (for image processing)
- **Dataset**: PlantVillage dataset (or custom dataset)

---

## 🛠️ Project Structure


---

## 🧪 Workflow

1. **Image Preprocessing**
   - Resize, normalize, and extract features from leaf images
2. **Feature Extraction**
   - Color histograms, shape, or texture-based features
3. **Model Training**
   - Trained using algorithms like Random Forest, SVM, or KNN
4. **Model Evaluation**
   - Accuracy, Precision, Recall, Confusion Matrix
5. **Prediction Interface**
   - Upload image → Predict disease label

---

## ✅ Results

- Achieved **accuracy of ~90%** using traditional ML models on preprocessed image data.
- SVM and Random Forest performed best for classification tasks.

---

## 🚀 How to Run

1. Clone the repo  
```bash
git clone https://github.com/YourUsername/plant_diseases_prediction.git
cd plant_diseases_prediction
Install dependencies

bash
pip install -r requirements.txt
Run training or prediction notebook

bash
jupyter notebook notebooks/train_model.ipynb
(Optional) Launch the Web App (if Flask/Streamlit is used)

ba
Edit
streamlit run app/app.py

📸 Sample Predictions
Input Leaf Image	Predicted Disease
Tomato leaf	Tomato Leaf Curl
Potato leaf	Early Blight

🙌 Acknowledgments
Dataset: PlantVillage Dataset

📬 Contact
Sarvesh Yashwant Redekar
📧 sarveshredekar9@gmail.com
