# 🧠 Skin Disease Detection using Deep Learning

A deep learning-based web application that classifies skin lesions into multiple disease categories using **CNN + Transfer Learning (EfficientNetB3)** and provides **explainability using Grad-CAM**.

---

## 🚀 Project Overview

Millions of people lack access to dermatologists, especially in rural areas. This project aims to bridge that gap by building an AI-powered system that can:

* Analyze skin lesion images
* Predict the disease category
* Provide confidence score
* Highlight important regions using Grad-CAM

---

## 🎯 Features

* ✅ Multi-class skin disease classification (7 classes)
* ✅ Transfer Learning (EfficientNetB3)
* ✅ Handles class imbalance using class weights
* ✅ Grad-CAM visualization (Explainable AI 🔥)
* ✅ Streamlit web app for real-time prediction
* ✅ Confidence-based prediction output

---

## 🧬 Dataset

* **HAM10000 Dataset (Human Against Machine)**
* Contains ~10,000 dermatoscopic images

### Classes:

* Actinic Keratoses (`akiec`)
* Basal Cell Carcinoma (`bcc`)
* Benign Keratosis (`bkl`)
* Dermatofibroma (`df`)
* Melanoma (`mel`)
* Melanocytic Nevi (`nv`)
* Vascular Lesions (`vasc`)

📌 *Dataset not included due to size constraints.*

---

## 🛠️ Tech Stack

| Tool               | Purpose            |
| ------------------ | ------------------ |
| Python             | Core programming   |
| TensorFlow / Keras | Model training     |
| EfficientNetB3     | Transfer learning  |
| OpenCV / PIL       | Image processing   |
| Streamlit          | Web app deployment |
| Scikit-learn       | Evaluation metrics |

---

## 📁 Project Structure

```
skin-disease-detection/
│
├── archive/          # Raw dataset (not included)
├── data/             # Processed train/test data (not included)
├── models/           # Trained model (not included)
│
├── utils/
│   ├── gradcam.py
│   └── preprocess.py
│
├── train.py
├── predict.py
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/skin-disease-detection.git
cd skin-disease-detection
```

---

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Download Dataset

Download HAM10000 from Kaggle and place inside:

```
archive/
```

---

### 5. Prepare Data

```bash
python prepare_data.py
```

---

### 6. Train Model

```bash
python train.py
```

---

### 7. Run Prediction

```bash
python predict.py
```

---

### 8. Launch Web App

```bash
streamlit run app.py
```

---

## 📊 Model Workflow

1. Data preprocessing (resize, normalize, augment)
2. Handle class imbalance using class weights
3. Transfer Learning using EfficientNetB3
4. Fine-tuning last layers
5. Evaluation using accuracy & confusion matrix
6. Deployment using Streamlit

---

## 🔥 Grad-CAM Visualization

Grad-CAM highlights the region of the image the model focuses on while making predictions.

👉 Helps improve trust in AI decisions (important in healthcare)

---

## 📈 Results

* Accuracy: ~80–90% (depending on tuning)
* Strong performance on common classes
* Lower confidence on minority classes due to imbalance

---

## ⚠️ Limitations

* Dataset is imbalanced
* Some classes have very similar visual features
* Not a substitute for professional medical diagnosis

---

## 🧾 Disclaimer

⚠️ This project is for **educational purposes only** and should **not be used for medical diagnosis**.

---

## 🧠 Future Improvements

* Improve accuracy using Focal Loss
* Add more datasets (ISIC, DermNet)
* Deploy on cloud (AWS / Render)
* Convert to mobile app
* Add top-3 prediction probabilities

---

## 👨‍💻 Author

**Dhruv Radadiya**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
