mport streamlit as st
import torch
import cv2
import numpy as np

# Load your segmentation model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'best_model.pt'
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load on CPU if GPU not available
    model.eval()
    return model

def preprocess_image(image):
    # Implement your image preprocessing logic here
    return image

def perform_segmentation(model, image):
    image = preprocess_image(image)
    with torch.no_grad():
        logits_mask = model(torch.Tensor(image).unsqueeze(0))
        pred_mask = torch.sigmoid(logits_mask)
        pred_mask = (pred_mask > 0.5) * 255.0  # Thresholding for binary mask
    return pred_mask.squeeze(0)

# Define Streamlit app
def main():
    st.title('Image Segmentation App')

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        model = load_model()
        image = cv2.imread(uploaded_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pred_mask = perform_segmentation(model, image)

        # Display original image and segmented mask
        st.image(image, caption='Original Image', use_column_width=True)
        st.image(pred_mask, caption='Segmented Mask', use_column_width=True)

if __name__ == '__main__':
    main()
