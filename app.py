import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from srcnn import SRCNN
from srgan import SRGANGenerator

def load_model(model_name: str):
    if model_name == "SRCNN":
        model_path = "models/srcnn/srcnn_final.pt"
        model = SRCNN()
    else:
        model_path = "models/srgan/srgan_generator_final.pt"
        model = SRGANGenerator()

    loaded_obj = torch.load(model_path, map_location="cpu")

    if isinstance(loaded_obj, dict):
        model.load_state_dict(loaded_obj)
    else:
        model = loaded_obj
        model.eval()

        if not hasattr(model, "layers"):
            try:
                rebuilt = SRCNN()
                rebuilt.load_state_dict(model.state_dict())
                model = rebuilt
            except Exception:
                raise RuntimeError(
                    "Loaded model object doesn't match current SRCNN definition. "
                    "Please re-save the model using state_dict (torch.save(model.state_dict(), path))."
                )

    model.eval()
    return model


def preprocess_image(image: Image.Image, scale_factor=4):
    w, h = image.size
    upsampled = image.resize((w * scale_factor, h * scale_factor), Image.BICUBIC)
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(upsampled).unsqueeze(0)


def postprocess_image(tensor):
    tensor = tensor.squeeze(0).detach().clamp(0, 1)
    return transforms.ToPILImage()(tensor)


def enhance_image(model, input_image):
    with torch.no_grad():
        return model(input_image)


st.set_page_config(page_title="Super-Resolution", page_icon="🔍", layout="centered")

st.title("Image Super-Resolution")
st.write("Upload a low-resolution image and enhance it using SRCNN or SRGAN.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("Choose Model", ["SRCNN", "SRGAN"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Enhance Image"):
        with st.spinner("Enhancing image..."):
            model = load_model(model_choice)
            input_tensor = preprocess_image(image)
            output_tensor = enhance_image(model, input_tensor)
            enhanced_image = postprocess_image(output_tensor)

        st.success("Enhancement complete!")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width='stretch')
        with col2:
            st.image(enhanced_image, caption=f"Enhanced Image ({model_choice})", width='stretch')

        enhanced_image.save("enhanced_output.png")
        with open("enhanced_output.png", "rb") as f:
            st.download_button(
                "Download Enhanced Image",
                f,
                file_name="enhanced_output.png",
                mime="image/png"
            )
    else:
        st.image(image, caption="Uploaded Image Preview", width='stretch')