---
title: TruthLens - Fake Image Detector
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: true
license: mit
---

# 🔍 TruthLens - AI-Powered Image Authenticity Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.10+-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

**TruthLens** is a deep learning-powered web application that detects fake or AI-generated images using Convolutional Neural Networks (CNN) and Gabor filter texture analysis.

## ✨ Features

- 🧠 **Deep Learning** - CNN-based classification for accurate detection
- 🔬 **Gabor Filters** - Advanced texture analysis for better accuracy
- ⚡ **Fast Results** - Get predictions in seconds
- 📊 **Confidence Score** - Detailed probability breakdown
- 🎨 **Beautiful UI** - Modern, responsive interface

## 🚀 Live Demo

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Bhanucreator/TruthLens)

## 📁 Project Structure

```
📦 ImageClasification
├── 📓 1_Process_Image.ipynb      # Data preprocessing with Gabor filters
├── 📓 CNN_FakeRealClasification.ipynb  # Model training notebook
├── 📓 InferenceFile.ipynb        # Testing and inference
├── 🐍 app.py                     # Streamlit web application
├── 🐍 utils.py                   # Utility functions
├── 📋 requirements.txt           # Python dependencies
├── 📁 Data/                      # Raw dataset (Fake/Real images)
├── 📁 dataset/                   # Processed data (data.npz)
├── 📁 model/                     # Saved model weights
├── 📁 processed/                 # Gabor filtered images
└── 📁 user_input/                # Images for testing
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/Bhanucreator/ImageClasifier.git
cd ImageClasifier

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📖 How It Works

### 1. Preprocessing Pipeline
- Raw images are processed using **Gabor filters** for texture extraction
- Images are resized to **128x128** pixels
- Pixel values are normalized (0-1)

### 2. Model Architecture
```
Input (128x128x3)
    ↓
Conv2D(70) + MaxPooling + BatchNorm
    ↓
Conv2D(20) + MaxPooling + BatchNorm
    ↓
Conv2D(10) + MaxPooling + BatchNorm
    ↓
Dropout(0.6)
    ↓
Flatten → Dense(512) → Dropout(0.3)
    ↓
Output: Dense(2) - Fake/Real
```

### 3. Training Notebooks

1. **Process the Data:** Run `1_Process_Image.ipynb` to preprocess images
2. **Train the Model:** Run `CNN_FakeRealClasification.ipynb` to train the CNN
3. **Test:** Run `InferenceFile.ipynb` or use the web app

## 🎯 Usage

### Web Application
1. Upload an image (JPG, PNG, WEBP)
2. Click "Analyze Image"
3. View results with confidence score

### Jupyter Notebooks
```python
# Load model and predict
from tensorflow.keras.models import load_model
model = load_model("model/cnn_model_weights.h5")
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | CNN (3 Conv layers) |
| Input Size | 128 × 128 × 3 |
| Optimizer | Adam (lr=0.0001) |
| Loss | Binary Crossentropy |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

**Bhanucreator**

- GitHub: [@Bhanucreator](https://github.com/Bhanucreator)
- Email: bhanukiran90216@gmail.com

---

<p align="center">
  Made with ❤️ using TensorFlow & Streamlit
</p>
