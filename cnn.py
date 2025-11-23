import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("bird_drone_classifier.h5")

IMG_SIZE = 224

st.title("Aerial Object Classifier (Bird vs Drone)")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    label = "Drone" if pred > 0.5 else "Bird"
    confidence = pred if pred > 0.5 else (1 - pred)

    st.write(f"### Prediction: **{label}**")
    st.write(f"### Confidence: **{confidence:.3f}**")
