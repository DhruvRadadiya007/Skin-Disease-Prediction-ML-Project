import tensorflow as tf
import numpy as np
from utils.preprocess import preprocess_image
import config

model = tf.keras.models.load_model(config.MODEL_PATH)

def predict(img_path):
    img = preprocess_image(img_path, config.IMG_SIZE)
    preds = model.predict(img)

    idx = np.argmax(preds)
    return config.CLASS_NAMES[idx], float(np.max(preds))


# -------------------------
# 🔥 ADD THIS PART
# -------------------------
if __name__ == "__main__":
    img_path = "sample.jpg"   # make sure this file exists

    result, confidence = predict(img_path)

    print("\n✅ Prediction:", result)
    print("🔥 Confidence:", confidence)