"""
🔍 TruthLens - AI-Powered Image Authenticity Detector
Detect fake/AI-generated images with advanced deep learning and Gabor filter texture analysis.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
import time
import os

from utils import (
    preprocess_image, 
    load_image_from_pil,
    get_confidence_color,
    get_confidence_emoji,
    format_confidence,
    apply_gabor_filter
)

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="TruthLens | Fake Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR AMAZING UI
# ============================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Upload Box */
    .upload-box {
        background: linear-gradient(145deg, #f0f2f6 0%, #e6e9ef 100%);
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #764ba2;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    /* Result Cards */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .result-real {
        background: linear-gradient(135deg, #00D26A 0%, #00B359 100%);
        color: white;
    }
    
    .result-fake {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF3333 100%);
        color: white;
    }
    
    /* Confidence Meter */
    .confidence-meter {
        background: #f0f2f6;
        border-radius: 50px;
        height: 30px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 50px;
        transition: width 1s ease-in-out;
    }
    
    /* Stats Cards */
    .stat-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        height: 100%;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styling */
    .sidebar-info {
        background: linear-gradient(145deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom Button */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_classification_model():
    """Load the trained CNN model."""
    model_path = "model/cnn_model_weights.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error("⚠️ Model file not found! Please ensure 'model/cnn_model_weights.h5' exists.")
        return None


# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_image(model, image: np.ndarray):
    """
    Make prediction on the input image.
    
    Returns:
        tuple: (prediction_label, confidence, raw_predictions)
    """
    # Preprocess image
    processed = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed, verbose=0)
    
    # Get results
    pred_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_class])
    
    labels = {0: "Fake", 1: "Real"}
    
    return labels[pred_class], confidence, predictions[0]


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def create_confidence_gauge(confidence: float, is_real: bool):
    """Create a beautiful gauge chart for confidence."""
    color = "#00D26A" if is_real else "#FF4B4B"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#ddd"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ddd",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff3e0'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Poppins'}
    )
    
    return fig


def create_probability_chart(predictions):
    """Create a bar chart showing class probabilities."""
    labels = ['Fake', 'Real']
    colors = ['#FF4B4B', '#00D26A']
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=predictions * 100,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in predictions],
            textposition='auto',
            textfont=dict(size=16, color='white')
        )
    ])
    
    fig.update_layout(
        title=dict(text='Class Probabilities', font=dict(size=18)),
        yaxis_title='Probability (%)',
        yaxis_range=[0, 100],
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Poppins'}
    )
    
    return fig


# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 TruthLens</h1>
        <p>AI-Powered Image Authenticity Detector</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 About TruthLens")
        st.markdown("""
        <div class="sidebar-info">
            <p><strong>TruthLens</strong> uses advanced deep learning to detect fake or AI-generated images.</p>
            <p>Our CNN model analyzes texture patterns using Gabor filters to identify manipulated content.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🛠️ How It Works")
        st.markdown("""
        1. 📤 **Upload** your image
        2. 🔬 **Gabor Filter** extracts textures
        3. 🧠 **CNN Model** analyzes patterns
        4. ✅ **Get Results** with confidence score
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Info")
        st.markdown("""
        - **Architecture:** CNN (3 Conv layers)
        - **Input Size:** 128 × 128 × 3
        - **Preprocessing:** Double Gabor Filter
        - **Classes:** Fake / Real
        """)
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.markdown("Made with ❤️ by **Bhanucreator**")
        st.markdown("[GitHub](https://github.com/Bhanucreator/ImageClasifier)")
    
    # Main Content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Supported formats: JPG, JPEG, PNG, WEBP"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)
            
            # Show image info
            st.markdown(f"""
            <div class="stat-card" style="margin-top: 1rem;">
                <p style="margin: 0; color: #666;">
                    <strong>File:</strong> {uploaded_file.name}<br>
                    <strong>Size:</strong> {image.size[0]} × {image.size[1]} px<br>
                    <strong>Format:</strong> {image.format or 'Unknown'}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyze button
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_clicked = st.button("🔍 Analyze Image", use_container_width=True)
        else:
            st.markdown("""
            <div class="upload-box">
                <p style="font-size: 3rem; margin: 0;">📁</p>
                <p style="font-size: 1.2rem; color: #666; margin: 0.5rem 0;">
                    Drag and drop your image here
                </p>
                <p style="color: #999; font-size: 0.9rem;">
                    or click to browse
                </p>
            </div>
            """, unsafe_allow_html=True)
            analyze_clicked = False
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if uploaded_file is not None and analyze_clicked:
            # Load model
            model = load_classification_model()
            
            if model is not None:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Loading
                status_text.text("🔄 Loading image...")
                progress_bar.progress(20)
                time.sleep(0.3)
                
                # Step 2: Preprocessing
                status_text.text("🔬 Applying Gabor filters...")
                progress_bar.progress(50)
                cv_image = load_image_from_pil(image)
                time.sleep(0.3)
                
                # Step 3: Prediction
                status_text.text("🧠 Analyzing with CNN...")
                progress_bar.progress(80)
                prediction, confidence, raw_preds = predict_image(model, cv_image)
                time.sleep(0.3)
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.2)
                progress_bar.empty()
                status_text.empty()
                
                # Results
                is_real = prediction == "Real"
                emoji = get_confidence_emoji(confidence, is_real)
                color = get_confidence_color(confidence, is_real)
                
                # Result Card
                result_class = "result-real" if is_real else "result-fake"
                st.markdown(f"""
                <div class="result-card {result_class}" style="text-align: center;">
                    <p style="font-size: 4rem; margin: 0;">{emoji}</p>
                    <h2 style="font-size: 2.5rem; margin: 0.5rem 0;">{prediction.upper()}</h2>
                    <p style="font-size: 1.5rem; opacity: 0.9;">
                        Confidence: {format_confidence(confidence)}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge Chart
                st.plotly_chart(
                    create_confidence_gauge(confidence, is_real),
                    use_container_width=True
                )
                
                # Probability Chart
                st.plotly_chart(
                    create_probability_chart(raw_preds),
                    use_container_width=True
                )
                
                # Show processed image
                with st.expander("🔬 View Gabor Filtered Image"):
                    gabor_img = apply_gabor_filter(cv_image)
                    gabor_rgb = cv2.cvtColor(gabor_img, cv2.COLOR_BGR2RGB)
                    st.image(gabor_rgb, caption="Gabor Filtered (Texture Analysis)", use_container_width=True)
                
        else:
            # Placeholder when no image
            st.markdown("""
            <div class="result-card" style="text-align: center; padding: 4rem 2rem;">
                <p style="font-size: 4rem; margin: 0; opacity: 0.3;">🔍</p>
                <p style="color: #999; font-size: 1.1rem;">
                    Upload an image and click <strong>Analyze</strong> to see results
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("---")
    st.markdown("### ✨ Features")
    
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
    
    with feat_col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h4>Deep Learning</h4>
            <p style="color: #666; font-size: 0.9rem;">
                CNN-based classification for accurate detection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h4>Gabor Filters</h4>
            <p style="color: #666; font-size: 0.9rem;">
                Advanced texture analysis for better accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4>Fast Results</h4>
            <p style="color: #666; font-size: 0.9rem;">
                Get predictions in seconds
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h4>Confidence Score</h4>
            <p style="color: #666; font-size: 0.9rem;">
                Detailed probability breakdown
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #999;">
        <p>🔍 <strong>TruthLens</strong> - Detecting fake images with AI</p>
        <p style="font-size: 0.8rem;">Made with ❤️ using Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
