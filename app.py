import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from labels_utils import load_labels

MODEL_PATH = "models/garbage_model.h5"

st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("Garbage Classifier")

if not st.sidebar.button("Reload model"):
    pass

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Model not found. First run: python train_model.py")
    st.stop()

inp_shape = model.input_shape
IMG_SIZE = inp_shape[1] if len(inp_shape) == 4 else 224

labels = load_labels("models/labels.txt")
if labels is None:
    labels = ["cardboard","glass","metal","paper","plastic","trash"]

threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4, 0.01)

def preprocess(img, size):
    img = img.convert("RGB").resize((size,size))
    arr = np.asarray(img, dtype="float32") / 255.0
    return np.expand_dims(arr, 0)

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, use_column_width=True)
    x = preprocess(img, IMG_SIZE)
    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    label = labels[idx] if idx < len(labels) else str(idx)
    if conf < threshold:
        st.error(f"Unknown (confidence {conf:.3f} < {threshold:.2f})")
    else:
        st.success(f"{label} ({conf:.3f})")