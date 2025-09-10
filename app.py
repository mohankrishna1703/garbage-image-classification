"""
app.py - Garbage Classifier with Improved Unknown Handling
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title
st.title("♻️ Garbage Image Classifier")

# Load model
MODEL_PATH = "models/garbage_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(predictions)

    # Adjusted threshold (40%)
    if confidence < 0.4:
        st.error("Prediction: Unknown 🚫")
    else:
        st.success(f"Prediction: {predicted_class}")