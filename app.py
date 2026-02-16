import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.h5")

# Class labels
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image to predict tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction")
    st.success(f"{class_names[predicted_class]} (Confidence: {confidence:.2f})")
