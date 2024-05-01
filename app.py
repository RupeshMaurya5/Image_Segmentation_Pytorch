import streamlit as st
import torch
import cv2
import numpy as np

# Define your model architecture and load weights
from segmentation_model import SegmentationModel
model = SegmentationModel()
model.load_state_dict(torch.load('path_to_your_trained_model'))
model.eval()

st.title('Image Segmentation App')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = cv2.imread(uploaded_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)  # Perform preprocessing as per your training pipeline

    # Perform segmentation
    with torch.no_grad():
        logits_mask = model(torch.Tensor(image).unsqueeze(0))
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 255.0  # Thresholding for binary mask

    # Display original image and segmented mask
    st.image(image, caption='Original Image', use_column_width=True)
    st.image(pred_mask.squeeze(0), caption='Segmented Mask', use_column_width=True)
