import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

model = load_model('best_model.pt')

st.title('Image Segmentation App')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for model input (resize, normalize, etc.)
    processed_image = preprocess_image(image)

    # Make prediction using your model
    segmented_image = model.predict(processed_image)

    # Display the segmented image
    st.image(segmented_image, caption='Segmented Image', use_column_width=True)
