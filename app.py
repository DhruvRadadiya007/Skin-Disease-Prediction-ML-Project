import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile

from utils.preprocess import preprocess_image
from utils.gradcam import get_gradcam, overlay_heatmap
import config

model = tf.keras.models.load_model(config.MODEL_PATH)

st.title("🧠 Skin Disease Detection")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(temp.name)

    img_array = preprocess_image(temp.name)

    preds = model.predict(img_array)
    idx = np.argmax(preds)

    disease = config.CLASS_NAMES[idx]
    confidence = np.max(preds) * 100

    st.subheader("Prediction")
    st.write(f"**Disease:** {disease}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Grad-CAM
    heatmap = get_gradcam(model, img_array)
    cam_img = overlay_heatmap(temp.name, heatmap)

    st.subheader("Grad-CAM Visualization")
    st.image(cam_img, clamp=True)

    if confidence < 60:
        st.warning("Low confidence prediction. Please consult a doctor.")

st.caption("⚠️ This is not a medical diagnosis tool.")