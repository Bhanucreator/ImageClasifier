"""
TruthLens - Utility Functions
Preprocessing and helper functions for Fake/Real Image Classification
"""

import cv2
import numpy as np
from PIL import Image
import io

# Image Configuration
IMG_SIZE = 128
CHANNELS = 3

# Gabor Filter Parameters
GABOR_KSIZE = (18, 18)
GABOR_SIGMA = 1.5
GABOR_THETA = np.pi / 4
GABOR_LAMBD = 5.0
GABOR_GAMMA = 1.5
GABOR_PSI = 0


def apply_gabor_filter(image: np.ndarray) -> np.ndarray:
    """
    Apply double Gabor filter for texture analysis.
    This extracts texture features that help distinguish fake from real images.
    
    Args:
        image: Input image as numpy array (BGR format)
    
    Returns:
        Gabor filtered image
    """
    # First Gabor filter
    gabor_1 = cv2.getGaborKernel(
        GABOR_KSIZE, GABOR_SIGMA, GABOR_THETA, 
        GABOR_LAMBD, GABOR_GAMMA, GABOR_PSI, 
        ktype=cv2.CV_32F
    )
    filtered_img_1 = cv2.filter2D(image, cv2.CV_8UC3, gabor_1)
    
    # Second Gabor filter (applied on first filtered image)
    gabor_2 = cv2.getGaborKernel(
        GABOR_KSIZE, GABOR_SIGMA, GABOR_THETA, 
        GABOR_LAMBD, GABOR_GAMMA, GABOR_PSI, 
        ktype=cv2.CV_32F
    )
    filtered_img_2 = cv2.filter2D(filtered_img_1, cv2.CV_8UC3, gabor_2)
    
    return filtered_img_2


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for a single image.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV)
    
    Returns:
        Preprocessed image ready for model prediction
    """
    # Apply Gabor filter
    filtered_img = apply_gabor_filter(image)
    
    # Resize to model input size
    resized_img = cv2.resize(filtered_img, (IMG_SIZE, IMG_SIZE))
    
    # Reshape for model input
    processed = np.reshape(resized_img, (IMG_SIZE, IMG_SIZE, CHANNELS))
    
    # Normalize pixel values
    processed = processed.astype(np.float32) / 255.0
    
    # Add batch dimension
    processed = np.expand_dims(processed, axis=0)
    
    return processed


def load_image_from_upload(uploaded_file) -> np.ndarray:
    """
    Load image from Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Image as numpy array in BGR format
    """
    # Read file bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    return image


def load_image_from_pil(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format.
    
    Args:
        pil_image: PIL Image object
    
    Returns:
        Image as numpy array in BGR format
    """
    # Convert PIL to numpy array (RGB)
    rgb_image = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    return bgr_image


def get_confidence_color(confidence: float, is_real: bool) -> str:
    """
    Get color based on prediction confidence.
    
    Args:
        confidence: Confidence score (0-1)
        is_real: Whether prediction is Real
    
    Returns:
        Hex color code
    """
    if is_real:
        if confidence > 0.8:
            return "#00D26A"  # Bright green
        elif confidence > 0.6:
            return "#7DD87D"  # Light green
        else:
            return "#FFA500"  # Orange (uncertain)
    else:
        if confidence > 0.8:
            return "#FF4B4B"  # Bright red
        elif confidence > 0.6:
            return "#FF7B7B"  # Light red
        else:
            return "#FFA500"  # Orange (uncertain)


def get_confidence_emoji(confidence: float, is_real: bool) -> str:
    """
    Get emoji based on prediction result.
    
    Args:
        confidence: Confidence score (0-1)
        is_real: Whether prediction is Real
    
    Returns:
        Emoji string
    """
    if is_real:
        if confidence > 0.8:
            return "✅"
        elif confidence > 0.6:
            return "🟢"
        else:
            return "🟡"
    else:
        if confidence > 0.8:
            return "🚨"
        elif confidence > 0.6:
            return "🔴"
        else:
            return "🟡"


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string."""
    return f"{confidence * 100:.1f}%"
