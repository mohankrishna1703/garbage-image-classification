import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from preprocess import create_generators

MODEL_PATH = "models/garbage_model.h5"
IMG_SIZE = 224
BATCH_SIZE = 16

def evaluate():
    val_flow = create_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE)[1]
    model = load_model(MODEL_PATH)
    preds = model.predict(val_flow, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_flow.classes
    inv = {v:k for k,v in val_flow.class_indices.items()}
    class_names = [inv[i] for i in range(len(inv))]
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

if __name__ == "__main__":
    evaluate()