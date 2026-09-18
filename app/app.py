"""
app.py
------
A simple web interface (Streamlit) so you (or anyone) can:
  1. Upload a product image
  2. See the model's reconstruction of it
  3. See a heatmap of WHERE it thinks the anomaly is
  4. Get a verdict: "OK" or "DEFECTIVE" with a confidence score

Run locally:
    streamlit run app/app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import ConvAutoencoder
from dataset import default_transform

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "autoencoder_best.pth")
THRESHOLD = 0.003618  # <-- your latest threshold from evaluate.py


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


def predict(image: Image.Image, model, device):
    tensor = default_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(tensor)
        error_map = ((tensor - recon) ** 2).mean(dim=1).squeeze().cpu().numpy()
        score = float(np.percentile(error_map, 99))

    input_img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    recon_img = recon.squeeze().permute(1, 2, 0).cpu().numpy()
    return input_img, recon_img, error_map, score


st.set_page_config(page_title="Industrial Anomaly Detector", layout="wide")
st.title("🔍 Unsupervised Anomaly Detection — Industrial Inspection")
st.write(
    "Upload a product image. The model was trained ONLY on normal/defect-free "
    "samples, so it flags anything it can't reconstruct well as an anomaly."
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    model, device = load_model()
    image = Image.open(uploaded_file).convert("RGB")

    input_img, recon_img, error_map, score = predict(image, model, device)

    verdict = "🔴 DEFECTIVE" if score > THRESHOLD else "🟢 OK (Normal)"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Input")
        st.image(input_img, use_container_width=True)
    with col2:
        st.subheader("Reconstruction")
        st.image(recon_img, use_container_width=True)
    with col3:
        st.subheader("Anomaly Heatmap")
        fig, ax = plt.subplots()
        ax.imshow(error_map, cmap="hot")
        ax.axis("off")
        st.pyplot(fig)

    st.markdown(f"### Verdict: {verdict}")
    st.write(f"Anomaly score: `{score:.6f}`  |  Threshold: `{THRESHOLD}`")
else:
    st.info("Waiting for an image to be uploaded...")
