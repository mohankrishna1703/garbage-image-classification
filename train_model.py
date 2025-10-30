import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from preprocess import create_generators
from labels_utils import save_labels

DATA_DIR = "data"
MODELS_DIR = "models"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 6

def build_and_train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_flow, val_flow = create_generators(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    inv = {v:k for k,v in train_flow.class_indices.items()}
    labels = [inv[i] for i in range(len(inv))]
    save_labels(labels, os.path.join(MODELS_DIR, "labels.txt"))
    print("Saved labels:", labels)

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3), pooling="avg")
    base.trainable = False

    x = base.output
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(train_flow.num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=outputs)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    model_file = os.path.join(MODELS_DIR, "garbage_model.h5")
    checkpoint = ModelCheckpoint(model_file, monitor="val_accuracy", save_best_only=True, verbose=1)
    model.fit(train_flow, validation_data=val_flow, epochs=EPOCHS, callbacks=[checkpoint])
    print("Training done. Model saved to", model_file)

if __name__ == "__main__":
    build_and_train()