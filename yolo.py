import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2

st.title("YOLO Bird & Drone Detector")
st.write("Upload an image and the model will detect Bird / Drone!")

# Load model
model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save input image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getvalue())
        img_path = tmp.name

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # YOLO inference
    results = model(img_path)
    result_img = results[0].plot()

    # Convert BGR to RGB
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    st.write("### Detection Result:")
    st.image(result_img, use_column_width=True)
