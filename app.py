import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import json
import numpy as np

st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020817 0%, #0f172a 50%, #111827 100%);
    color: white;
}

h1 {
    text-align: center;
    font-size: 52px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 30px;
}

section[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.85);
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #334155;
    box-shadow: 0 0 20px rgba(0,0,0,0.25);
}

img {
    border-radius: 18px;
    border: 2px solid #334155;
    box-shadow: 0 0 30px rgba(56, 189, 248, 0.18);
}

.result-card {
    background: rgba(15, 23, 42, 0.9);
    padding: 26px;
    border-radius: 22px;
    border: 1px solid #334155;
    margin-top: 22px;
    box-shadow: 0 0 25px rgba(0,0,0,0.35);
}

.info-box {
    background-color: #020817;
    padding: 14px;
    border-radius: 12px;
    margin-top: 10px;
    border: 1px solid #1e293b;
}
.result-card {
    animation: fadeIn 0.6s ease;
}

img:hover {
    transform: scale(1.02);
    transition: 0.3s ease;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("classes.json") as f:
    classes = json.load(f)

display_names = {
    "glioma": "Glioma Tumor",
    "meningioma": "Meningioma Tumor",
    "pituitary": "Pituitary Tumor",
    "notumor": "No Tumor Detected"
}

model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Linear(256, len(classes))
)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

st.title("Brain Tumor Detection")

st.markdown(
    '<div class="subtitle">Upload a brain MRI image to classify the scan.</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    img_array = np.array(image)

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    color_difference = (
        np.abs(r - g).mean() +
        np.abs(r - b).mean() +
        np.abs(g - b).mean()
    )

    st.image(image, width=350)

    if color_difference > 60:
        st.error("Please upload a valid brain MRI image.")

    else:
        x = transform(image)
        x = x.unsqueeze(0).to(device)

        with st.spinner("Analyzing MRI scan..."):
            with torch.no_grad():
                out = model(x)
                probabilities = torch.softmax(out, dim=1)
                confidence, pred = torch.max(probabilities, 1)

        raw_class = classes[str(pred.item())]
        predicted_class = display_names.get(raw_class, raw_class)
        confidence_score = confidence.item() * 100

        if confidence_score < 85:
            st.error("Invalid or unclear image. Please upload a brain MRI scan.")

        else:
            if raw_class == "notumor":
                result_color = "#22c55e"
            else:
                result_color = "#38bdf8"

            st.markdown(f"""
            <div class="result-card">
            <h3 style="color:{result_color}; margin-bottom:18px;">
            Prediction Result
            </h3>

            <div style="background-color:#020817; padding:14px; border-radius:12px; margin-top:10px; border:1px solid #1e293b;">
            <b>Class:</b> {predicted_class}
            </div>

            <div style="background-color:#020817; padding:14px; border-radius:12px; margin-top:10px; border:1px solid #1e293b;">
            <b>Confidence:</b> {confidence_score:.2f}%
            </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(confidence_score))