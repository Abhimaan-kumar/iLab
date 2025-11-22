import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras import layers, models


# Paths
train_dir = "dataset/training"
val_dir = "dataset/validation"
test_dir = "dataset/testing"

img_size = (224, 224)
batch_size = 32


def create_validation_split(train_directory, val_directory, val_ratio=0.2, seed=42, copy=True):
    """Create a validation folder by copying a fraction of images from each class in train_directory.

    If `val_directory` already exists and contains files, this function does nothing.
    """
    train_directory = os.path.abspath(train_directory)
    val_directory = os.path.abspath(val_directory)

    # If validation directory already exists and is non-empty, skip
    if os.path.isdir(val_directory):
        # check for any image files
        for root, dirs, files in os.walk(val_directory):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    print(f"Validation directory '{val_directory}' already exists and contains images; skipping split creation.")
                    return

    random.seed(seed)

    # Walk classes in training directory
    for class_name in os.listdir(train_directory):
        class_train_path = os.path.join(train_directory, class_name)
        if not os.path.isdir(class_train_path):
            continue

        class_val_path = os.path.join(val_directory, class_name)
        os.makedirs(class_val_path, exist_ok=True)

        images = [f for f in os.listdir(class_train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
        if not images:
            continue

        n_val = max(1, int(len(images) * val_ratio)) if len(images) > 1 else 0
        selected = random.sample(images, n_val) if n_val > 0 else []

        for fname in selected:
            src = os.path.join(class_train_path, fname)
            dst = os.path.join(class_val_path, fname)
            try:
                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)
            except Exception as e:
                print(f"Failed copying {src} -> {dst}: {e}")


# Create validation split if needed
create_validation_split(train_dir, val_dir, val_ratio=0.2, seed=42, copy=True)

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# Get class names before prefetch
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Prefetch datasets
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Build model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=img_size + (3,))
x = layers.Rescaling(1.0 / 255)(inputs)  # simple normalization
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model: use validation dataset during training to improve model selection
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# After training, evaluate on the held-out test set
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# Save model in safest format
model.save("eye_disease_classifier.keras")
print("Model saved as eye_disease_classifier.keras")
