import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Dataset
data_dir = "data"
img_height, img_width = 128, 128
batch_size = 32

# Preprocessing (just rescaling + validation split)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Very simple CNN (beginner-level)
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(train_data.num_classes, activation="softmax")
])

# Compile
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
model.fit(train_data, validation_data=val_data, epochs=10)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/garbage_cnn.h5")

print("Training complete. Model saved in models/garbage_cnn.h5")