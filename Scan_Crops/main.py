import tensorflow as tf
import numpy as np
import pandas as pd

# Load model + class names
model = tf.keras.models.load_model("trained_crop_disease_model.h5")
class_df = pd.read_csv("class_names.csv")
class_names = class_df['class_name'].tolist()

def predict_disease(image_path):
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Convert grayscale → RGB
    if img_array.ndim == 2 or img_array.shape[-1] == 1:
        img_array = np.stack((img_array.squeeze(),)*3, axis=-1)

    img_array = np.expand_dims(img_array, 0) / 255.0

    # Prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]))

    return predicted_class, confidence

# 🔥 Example
img_path = "images.jpg"
disease, conf = predict_disease(img_path)
print(f"Predicted: {disease} ({conf:.2%} confidence)")
