import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title
st.title("Garbage Image Classifier")

# Load model
MODEL_PATH = "models/garbage_cnn.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    st.error("Model file not found! Please train the model first using train.py")
    st.stop()

# Class labels - same order as training
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((128, 128))  # match training size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.write(f"**Prediction:** {predicted_class}")