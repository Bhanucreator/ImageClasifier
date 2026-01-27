"""
🔍 TruthLens - AI-Powered Image Authenticity Detector
A modern web application for detecting fake/AI-generated images
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import time
import os
import base64
from datetime import datetime

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
# Load logo for favicon
logo_icon = Image.open("logo.png") if os.path.exists("logo.png") else "🔍"

st.set_page_config(
    page_title="TruthLens | AI Fake Image Detector",
    page_icon=logo_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# UTILITY FUNCTIONS
# ============================================
def get_logo_base64():
    """Load logo and convert to base64."""
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

def get_image_base64(image_path):
    """Convert image to base64."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

LOGO_BASE64 = get_logo_base64()

# ============================================
# MODERN CSS STYLING
# ============================================
st.markdown("""
<style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== CSS VARIABLES ===== */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a25;
        --bg-card-hover: #22222f;
        --accent-primary: #7c3aed;
        --accent-secondary: #a855f7;
        --accent-tertiary: #c084fc;
        --success: #22c55e;
        --success-light: #4ade80;
        --danger: #ef4444;
        --danger-light: #f87171;
        --warning: #eab308;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --border-color: rgba(124, 58, 237, 0.2);
        --border-hover: rgba(124, 58, 237, 0.5);
        --glow-purple: 0 0 40px rgba(124, 58, 237, 0.3);
        --glow-green: 0 0 40px rgba(34, 197, 94, 0.3);
        --glow-red: 0 0 40px rgba(239, 68, 68, 0.3);
    }
    
    /* ===== GLOBAL RESET ===== */
    .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-primary);
        background-image: 
            radial-gradient(ellipse at 10% 10%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 90%, rgba(168, 85, 247, 0.06) 0%, transparent 50%);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu, footer, header, .stDeployButton {display: none !important; visibility: hidden !important;}
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {width: 8px; height: 8px;}
    ::-webkit-scrollbar-track {background: var(--bg-secondary);}
    ::-webkit-scrollbar-thumb {background: var(--accent-primary); border-radius: 4px;}
    ::-webkit-scrollbar-thumb:hover {background: var(--accent-secondary);}
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f18 0%, #1a1a28 100%);
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    
    /* ===== NAVBAR / LOGO SECTION ===== */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 24px;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        margin-bottom: 24px;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    
    .nav-logo {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        object-fit: cover;
    }
    
    .nav-title {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    
    .nav-badge {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* ===== HERO SECTION ===== */
    .hero {
        text-align: center;
        padding: 48px 24px;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.12) 0%, rgba(168, 85, 247, 0.08) 100%);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: -100px;
        left: 50%;
        transform: translateX(-50%);
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(124, 58, 237, 0.2) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
    }
    
    .hero h1 {
        font-size: 52px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #e9d5ff 50%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 12px 0;
        letter-spacing: -1.5px;
    }
    
    .hero-subtitle {
        color: var(--text-secondary);
        font-size: 18px;
        font-weight: 400;
        margin: 0;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 48px;
        margin-top: 32px;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--accent-tertiary);
    }
    
    .stat-label {
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 4px;
    }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .section-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    
    /* ===== CARDS ===== */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: var(--border-hover);
        box-shadow: var(--glow-purple);
    }
    
    /* ===== UPLOAD ZONE ===== */
    .upload-zone {
        border: 2px dashed var(--border-hover);
        border-radius: 16px;
        padding: 48px 32px;
        text-align: center;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.05) 0%, rgba(168, 85, 247, 0.03) 100%);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-zone:hover {
        border-color: var(--accent-primary);
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(168, 85, 247, 0.06) 100%);
    }
    
    .upload-icon {
        font-size: 64px;
        margin-bottom: 16px;
    }
    
    .upload-title {
        color: var(--text-primary);
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .upload-subtitle {
        color: var(--text-muted);
        font-size: 14px;
    }
    
    /* ===== IMAGE INFO ===== */
    .image-info {
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid rgba(124, 58, 237, 0.1);
    }
    
    .info-row:last-child {
        border-bottom: none;
    }
    
    .info-label {
        color: var(--text-muted);
        font-size: 13px;
        font-weight: 500;
    }
    
    .info-value {
        color: var(--text-primary);
        font-size: 13px;
        font-weight: 600;
    }
    
    /* ===== RESULT CARDS ===== */
    .result-card {
        border-radius: 20px;
        padding: 40px 32px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    
    .result-real {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(74, 222, 128, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.3);
        box-shadow: var(--glow-green);
    }
    
    .result-real::before {
        background: linear-gradient(90deg, var(--success), var(--success-light));
    }
    
    .result-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(248, 113, 113, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: var(--glow-red);
    }
    
    .result-fake::before {
        background: linear-gradient(90deg, var(--danger), var(--danger-light));
    }
    
    .result-icon {
        font-size: 72px;
        margin-bottom: 16px;
    }
    
    .result-label {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 8px;
    }
    
    .result-real .result-label {
        color: var(--success);
    }
    
    .result-fake .result-label {
        color: var(--danger);
    }
    
    .result-confidence {
        font-size: 18px;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* ===== AWAITING RESULT ===== */
    .awaiting-result {
        background: var(--bg-card);
        border: 1px dashed var(--border-color);
        border-radius: 20px;
        padding: 64px 32px;
        text-align: center;
    }
    
    .awaiting-icon {
        font-size: 80px;
        opacity: 0.3;
        margin-bottom: 20px;
    }
    
    .awaiting-text {
        color: var(--text-muted);
        font-size: 16px;
    }
    
    /* ===== FEATURES GRID ===== */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 16px;
    }
    
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: var(--border-hover);
        box-shadow: var(--glow-purple);
    }
    
    .feature-icon {
        font-size: 36px;
        margin-bottom: 12px;
    }
    
    .feature-title {
        color: var(--text-primary);
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    
    .feature-desc {
        color: var(--text-muted);
        font-size: 12px;
        line-height: 1.5;
    }
    
    /* ===== SIDEBAR SECTIONS ===== */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 20px;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        margin-bottom: 24px;
    }
    
    .sidebar-logo img {
        width: 50px;
        height: 50px;
        border-radius: 12px;
    }
    
    .sidebar-logo-text h2 {
        font-size: 22px;
        font-weight: 800;
        background: linear-gradient(135deg, #fff, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .sidebar-logo-text span {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .sidebar-section {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .sidebar-section-title {
        color: var(--accent-tertiary);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .step-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 14px;
    }
    
    .step-num {
        width: 26px;
        height: 26px;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
    }
    
    .step-text {
        color: var(--text-secondary);
        font-size: 13px;
        line-height: 1.6;
    }
    
    .step-text strong {
        color: var(--text-primary);
    }
    
    .spec-grid {
        display: grid;
        gap: 10px;
    }
    
    .spec-item {
        display: flex;
        justify-content: space-between;
        padding: 10px 14px;
        background: rgba(124, 58, 237, 0.1);
        border-radius: 8px;
    }
    
    .spec-label {
        color: var(--text-muted);
        font-size: 12px;
    }
    
    .spec-value {
        color: var(--text-primary);
        font-size: 12px;
        font-weight: 600;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        padding: 32px 16px;
        margin-top: 48px;
        border-top: 1px solid var(--border-color);
    }
    
    .footer-brand {
        font-size: 18px;
        font-weight: 700;
        color: var(--accent-tertiary);
        margin-bottom: 8px;
    }
    
    .footer-text {
        color: var(--text-muted);
        font-size: 13px;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary)) !important;
        color: white !important;
        border: none !important;
        padding: 16px 32px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.3px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 24px -8px rgba(124, 58, 237, 0.5) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px -8px rgba(124, 58, 237, 0.6) !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    .stFileUploader > div {
        background: rgba(124, 58, 237, 0.05) !important;
        border: 2px dashed var(--border-hover) !important;
        border-radius: 12px !important;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--accent-primary) !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)) !important;
        border-radius: 10px;
    }
    
    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .hero h1 { font-size: 36px; }
        .features-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-stats { gap: 24px; }
        .nav-badge { display: none; }
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
    return None


# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_image(model, image: np.ndarray):
    """Make prediction on the input image."""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)
    pred_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][pred_class])
    labels = {0: "Fake", 1: "Real"}
    return labels[pred_class], confidence, predictions[0]


# ============================================
# VISUALIZATION
# ============================================
def create_confidence_gauge(confidence: float, is_real: bool):
    """Create a modern semicircle gauge chart."""
    color = "#22c55e" if is_real else "#ef4444"
    light_color = "rgba(34, 197, 94, 0.3)" if is_real else "rgba(239, 68, 68, 0.3)"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'suffix': "%", 
            'font': {'size': 56, 'color': color, 'family': 'Plus Jakarta Sans'},
            'valueformat': '.1f'
        },
        gauge={
            'axis': {
                'range': [0, 100], 
                'tickwidth': 2, 
                'tickcolor': "#27272a",
                'tickfont': {'color': '#52525b', 'size': 12},
                'dtick': 20
            },
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': "#18181b",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [25, 50], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(234, 179, 8, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(34, 197, 94, 0.2)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': confidence * 100
            }
        }
    ))
    
    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Plus Jakarta Sans'}
    )
    
    return fig


def create_probability_bars(predictions):
    """Create modern donut chart for probabilities."""
    labels = ['Fake', 'Real']
    colors = ['#ef4444', '#22c55e']
    values = [pred * 100 for pred in predictions]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(
            colors=colors,
            line=dict(color='#0a0a0f', width=2)
        ),
        textinfo='percent',
        textposition='inside',
        textfont=dict(size=14, color='white', family='Plus Jakarta Sans'),
        hovertemplate='<b>%{label}</b><br>Probability: %{value:.1f}%<extra></extra>',
        direction='clockwise',
        sort=False
    )])
    
    # Add center annotation with result
    dominant = 'Real' if values[1] > values[0] else 'Fake'
    dominant_value = max(values)
    dominant_color = '#22c55e' if values[1] > values[0] else '#ef4444'
    
    fig.add_annotation(
        text=f"<b>{dominant_value:.1f}%</b><br><span style='font-size:10px;color:#71717a'>{dominant}</span>",
        x=0.5, y=0.5,
        font=dict(size=18, color=dominant_color, family='Plus Jakarta Sans'),
        showarrow=False
    )
    
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=11, color='#a1a1aa')
        ),
        font={'family': 'Plus Jakarta Sans'}
    )
    
    return fig


# ============================================
# SESSION STATE
# ============================================
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    # ===== SIDEBAR =====
    with st.sidebar:
        # Logo Section
        if LOGO_BASE64:
            st.markdown(f"""
            <div class="sidebar-logo">
                <img src="data:image/png;base64,{LOGO_BASE64}" alt="Logo">
                <div class="sidebar-logo-text">
                    <h2>TruthLens</h2>
                    <span>AI Detection</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-logo">
                <div style="font-size: 40px;">🔍</div>
                <div class="sidebar-logo-text">
                    <h2>TruthLens</h2>
                    <span>AI Detection</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # How It Works
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">🛠️ How It Works</div>
            <div class="step-item">
                <span class="step-num">1</span>
                <span class="step-text"><strong>Upload</strong> your image file</span>
            </div>
            <div class="step-item">
                <span class="step-num">2</span>
                <span class="step-text"><strong>Gabor Filter</strong> extracts textures</span>
            </div>
            <div class="step-item">
                <span class="step-num">3</span>
                <span class="step-text"><strong>CNN Model</strong> analyzes patterns</span>
            </div>
            <div class="step-item">
                <span class="step-num">4</span>
                <span class="step-text"><strong>Get Results</strong> instantly</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Specifications
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-section-title">📊 Model Specs</div>
            <div class="spec-grid">
                <div class="spec-item">
                    <span class="spec-label">Architecture</span>
                    <span class="spec-value">CNN (3 Conv)</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Input Size</span>
                    <span class="spec-value">128 × 128</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Preprocessing</span>
                    <span class="spec-value">Dual Gabor</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Output</span>
                    <span class="spec-value">Fake / Real</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Session Stats
        st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">📈 Session Stats</div>
            <div class="spec-grid">
                <div class="spec-item">
                    <span class="spec-label">Images Analyzed</span>
                    <span class="spec-value">{st.session_state.total_analyzed}</span>
                </div>
                <div class="spec-item">
                    <span class="spec-label">Session Started</span>
                    <span class="spec-value">{datetime.now().strftime('%H:%M')}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 12px 0;">
            <p style="color: #a1a1aa; font-size: 12px; margin: 0;">Made with ❤️ by</p>
            <p style="color: #fff; font-size: 14px; font-weight: 600; margin: 4px 0;">Bhanucreator</p>
            <a href="https://github.com/Bhanucreator/ImageClasifier" target="_blank" 
               style="color: #a855f7; font-size: 11px; text-decoration: none;">
                View on GitHub →
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== MAIN CONTENT =====
    
    # Navbar
    if LOGO_BASE64:
        st.markdown(f"""
        <div class="navbar">
            <div class="nav-brand">
                <img src="data:image/png;base64,{LOGO_BASE64}" class="nav-logo" alt="Logo">
                <span class="nav-title">TruthLens</span>
            </div>
            <span class="nav-badge">AI Powered</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="navbar">
            <div class="nav-brand">
                <span style="font-size: 36px;">🔍</span>
                <span class="nav-title">TruthLens</span>
            </div>
            <span class="nav-badge">AI Powered</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero">
        <div class="hero-content">
            <h1>Detect Fake Images Instantly</h1>
            <p class="hero-subtitle">Powered by Deep Learning & Gabor Filter Texture Analysis</p>
            <div class="hero-stats">
                <div class="stat-item">
                    <div class="stat-value">CNN</div>
                    <div class="stat-label">Deep Learning</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">128px</div>
                    <div class="stat-label">Input Resolution</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">&lt;1s</div>
                    <div class="stat-label">Analysis Time</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    # LEFT COLUMN - Upload
    with col1:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📤</div>
            <h2 class="section-title">Upload Image</h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload",
            type=['jpg', 'jpeg', 'png', 'webp'],
            label_visibility="collapsed",
            help="Drag and drop or click to upload. Supports JPG, PNG, WEBP"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Display image with controlled size
            col_img1, col_img2, col_img3 = st.columns([1, 3, 1])
            with col_img2:
                st.image(image, caption="Uploaded Image", width=280)
            
            # Image Info Card
            st.markdown(f"""
            <div class="image-info">
                <div class="info-row">
                    <span class="info-label">📁 Filename</span>
                    <span class="info-value">{uploaded_file.name}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">📐 Dimensions</span>
                    <span class="info-value">{image.size[0]} × {image.size[1]} px</span>
                </div>
                <div class="info-row">
                    <span class="info-label">💾 File Size</span>
                    <span class="info-value">{uploaded_file.size / 1024:.1f} KB</span>
                </div>
                <div class="info-row">
                    <span class="info-label">🎨 Mode</span>
                    <span class="info-value">{image.mode}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)
        else:
            st.markdown("""
            <div class="upload-zone">
                <div class="upload-icon">🖼️</div>
                <p class="upload-title">Drop your image here</p>
                <p class="upload-subtitle">Supports JPG, JPEG, PNG, WEBP • Max 200MB</p>
            </div>
            """, unsafe_allow_html=True)
            analyze_btn = False
    
    # RIGHT COLUMN - Results
    with col2:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <h2 class="section-title">Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None and analyze_btn:
            model = load_classification_model()
            
            if model is not None:
                # Progress Animation
                progress = st.progress(0)
                status = st.empty()
                
                status.markdown("🔄 **Loading image...**")
                progress.progress(20)
                time.sleep(0.15)
                
                status.markdown("🔬 **Applying Gabor filters...**")
                progress.progress(45)
                cv_image = load_image_from_pil(image)
                time.sleep(0.15)
                
                status.markdown("🧠 **Running CNN analysis...**")
                progress.progress(70)
                prediction, confidence, raw_preds = predict_image(model, cv_image)
                time.sleep(0.15)
                
                status.markdown("✨ **Finalizing results...**")
                progress.progress(100)
                time.sleep(0.1)
                
                progress.empty()
                status.empty()
                
                # Update session stats
                st.session_state.total_analyzed += 1
                
                # Display Results
                is_real = prediction == "Real"
                emoji = get_confidence_emoji(confidence, is_real)
                result_class = "result-real" if is_real else "result-fake"
                
                st.markdown(f"""
                <div class="result-card {result_class}">
                    <div class="result-icon">{emoji}</div>
                    <div class="result-label">{prediction.upper()}</div>
                    <div class="result-confidence">Confidence: {format_confidence(confidence)}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge Chart
                st.plotly_chart(create_confidence_gauge(confidence, is_real), use_container_width=True)
                
                # Probability Donut Chart
                st.markdown("<p style='color: #a1a1aa; font-size: 14px; font-weight: 600; margin-bottom: 8px;'>📊 Probability Distribution</p>", unsafe_allow_html=True)
                st.plotly_chart(create_probability_bars(raw_preds), use_container_width=True)
                
                # Gabor Filter View
                with st.expander("🔬 View Gabor Filtered Image"):
                    gabor_img = apply_gabor_filter(cv_image)
                    gabor_rgb = cv2.cvtColor(gabor_img, cv2.COLOR_BGR2RGB)
                    gcol1, gcol2, gcol3 = st.columns([1, 2, 1])
                    with gcol2:
                        st.image(gabor_rgb, caption="Texture Analysis", width=220)
            else:
                st.error("⚠️ Model file not found. Please ensure `model/cnn_model_weights.h5` exists.")
        else:
            st.markdown("""
            <div class="awaiting-result">
                <div class="awaiting-icon">🔍</div>
                <p class="awaiting-text">Upload an image and click <strong>Analyze</strong> to see results</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ===== FEATURES SECTION =====
    st.markdown("---")
    st.markdown("""
    <div class="section-header" style="margin-top: 32px;">
        <div class="section-icon">✨</div>
        <h2 class="section-title">Key Features</h2>
    </div>
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Deep Learning</div>
            <div class="feature-desc">3-layer CNN architecture trained on thousands of images</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Gabor Filters</div>
            <div class="feature-desc">Advanced texture analysis for detecting manipulation artifacts</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Instant Results</div>
            <div class="feature-desc">Get predictions in under a second</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">High Accuracy</div>
            <div class="feature-desc">Reliable detection with confidence scores</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== FOOTER =====
    st.markdown("""
    <div class="footer">
        <div class="footer-brand">🔍 TruthLens</div>
        <p class="footer-text">AI-Powered Fake Image Detection • Built with Streamlit & TensorFlow</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
