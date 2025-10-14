import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

def load_model(model_name: str):
    """Load either SRCNN or SRGAN model."""
    if model_name == "SRCNN":
        model_path = "models/srcnn/srcnn_complete_model.pt"
    else:
        model_path = "models/srgan/srgan_generator_complete_model.pt"
    
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)


def postprocess_image(tensor):
    """Convert tensor back to PIL image."""
    tensor = tensor.squeeze(0).detach().clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    return image


def enhance_image(model, input_image):
    """Run model inference."""
    with torch.no_grad():
        output = model(input_image)
    return output


st.set_page_config(page_title="Super-Resolution GUI", page_icon="🔍", layout="centered")

st.title("Image Super-Resolution")
st.write("Upload a low-resolution image and choose a model to enhance it.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

model_choice = st.selectbox("Choose Model", ["SRCNN", "SRGAN"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Enhance Image"):
        with st.spinner("Loading model and enhancing image..."):
            model = load_model(model_choice)
            input_tensor = preprocess_image(image)
            output_tensor = enhance_image(model, input_tensor)
            enhanced_image = postprocess_image(output_tensor)

        st.success("Enhancement complete!")
        st.image(enhanced_image, caption=f"Enhanced Image ({model_choice})", use_container_width=True)
        
        # Download image
        enhanced_image.save("enhanced_output.png")
        with open("enhanced_output.png", "rb") as f:
            st.download_button("Download Enhanced Image", f, file_name="enhanced_output.png", mime="image/png")
