import tensorflow as tf
import numpy as np
import pandas as pd

# Load model and classes
model = tf.keras.models.load_model("Scan_Crops/trained_crop_disease_model.h5")
class_df = pd.read_csv("Scan_Crops/class_names.csv")
class_names = class_df['class_name'].tolist()

# Load full disease knowledge base
disease_db = pd.read_csv("Scan_Crops/crop_disease_database.csv")  # use your file name here

def predict_disease_with_info(image_path):
    """Predict crop disease and fetch details from knowledge base"""
    # Preprocess image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Handle grayscale → RGB
    if img_array.ndim == 2 or img_array.shape[-1] == 1:
        img_array = np.stack((img_array.squeeze(),)*3, axis=-1)
    img_array = np.expand_dims(img_array, 0) / 255.0

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions[0]))

    # Look up disease info in CSV
    match = disease_db[disease_db['Disease'] == predicted_class]
    if not match.empty:
        info = match.iloc[0].to_dict()
    else:
        info = {
            "Crop": "Unknown",
            "Description": "No information available.",
            "Symptoms": "N/A",
            "Organic_Remedy": "N/A",
            "Chemical_Remedy": "N/A",
            "Prevention": "N/A",
            "Season": "N/A",
            "Region": "N/A"
        }

    # Build result
    result = {
        "Predicted Disease": predicted_class,
        "Confidence": f"{confidence:.2%}",
        "Crop": info["Crop"],
        "Description": info["Description"],
        "Symptoms": info["Symptoms"],
        "Organic Remedy": info["Organic_Remedy"],
        "Chemical Remedy": info["Chemical_Remedy"],
        "Prevention": info["Prevention"],
        "Season": info["Season"],
        "Region": info["Region"]
    }
    return result


# 🔥 Example test
if __name__ == "__main__":
    test_img = "Scan_Crops/images.jpg"  
    res = predict_disease_with_info(test_img)
    for k, v in res.items():
        print(f"{k}: {v}")
