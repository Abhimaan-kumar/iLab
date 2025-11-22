import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load saved model
model = tf.keras.models.load_model("eye_disease_classifier.keras")

# Update class names exactly as in training
class_names = [
    'Cataract', 'Conjunctivitis', 'Eyelid', 'Jaundice',
    'Normal', 'Pterygium', 'Subconjunctival Hemorrage'
]

# Path to image
img_path = r"A:/iLab/dataset/training/Cataract/image_28.jpeg"

# Preprocess input image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # same normalization as training
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
pred_class = class_names[np.argmax(pred)]

print("Prediction:", pred_class)
