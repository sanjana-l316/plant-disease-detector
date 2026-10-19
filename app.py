import json
import torch
import torch.nn as nn
import streamlit as st
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

@st.cache_resource
def load_model():
    class_names = json.load(open("class_names.json"))
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load("plant_model.pth", map_location="cpu"))
    model.eval()
    return model, class_names

model, class_names = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def pretty(name):
    plant, condition = name.split("___")
    return f"{plant.replace('_', ' ')}: {condition.replace('_', ' ')}"

st.title("🌿 Plant Disease Detector")
st.write("Upload a clear photo of a single leaf and the model will diagnose it.")

file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Your upload", use_container_width=True)

    with torch.no_grad():
        probs = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
    top = torch.topk(probs, 3)

    best = pretty(class_names[top.indices[0]])
    conf = top.values[0].item()
    if conf >= 0.80:
        st.success(f"Prediction: **{best}** ({conf:.1%} confident)")
    else:
        st.warning(f"Not sure ({conf:.1%}). Best guess: **{best}**. "
                   "This model only knows 14 crops and works best on a single leaf on a plain background.")

    st.subheader("Top 3 guesses")
    for p, i in zip(top.values, top.indices):
        st.write(f"{pretty(class_names[i])}: {p.item():.1%}")
        st.progress(float(p))