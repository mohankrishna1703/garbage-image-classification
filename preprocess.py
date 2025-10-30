from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(data_dir="data", img_size=224, batch_size=16, val_split=0.2):
    train_gen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        validation_split=val_split
    )

    train_flow = train_gen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_flow = train_gen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )
    return train_flow, val_flow