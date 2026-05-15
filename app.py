import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("classes.json") as f:
    classes=json.load(f)

model=models.efficientnet_b0(weights=None)

model.classifier=nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features,256),
    nn.ReLU(),
    nn.Linear(256,len(classes))
)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

st.title("Brain Tumor Detection")

uploaded=st.file_uploader(
    "Upload MRI Image",
    type=["jpg","png","jpeg"]
)

if uploaded:

    image=Image.open(uploaded).convert("RGB")

    st.image(image,width=300)

    x=transform(image)
    x=x.unsqueeze(0).to(device)

    with torch.no_grad():
        out=model(x)
        pred=out.argmax(1).item()

    st.success(
        f"Prediction: {classes[str(pred)]}"
    )