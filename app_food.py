import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import requests
import os

model_path = "best_food2.pt"

# Download model if not available
if not os.path.exists(model_path):
    url = "https://huggingface.co/Sadrawi/FoodProf/resolve/main/best_food2.pt"
    with open(model_path, 'wb') as f:
        f.write(requests.get(url).content)

# Load YOLO segmentation model
model = YOLO(model_path)

st.title("YOLO Food Segmentation Demo")
st.write("Upload an image to segment and compare food area percentages relative to the plate.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="📤 Uploaded Image", use_container_width=True)

    if st.button("Run Segmentation"):
        # Convert to numpy array
        img_np = np.array(image)

        # Run YOLO prediction
        results = model.predict(img_np, conf=0.5)

        # Segmentation visualization
        seg_img = results[0].plot()  # overlay masks

        with col2:
            st.image(seg_img, caption="Segmentation Result", use_container_width=True)

        # Extract results
        r = results[0]
        masks = r.masks.data.cpu().numpy()   # (N, H, W)
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # Compute areas
        plate_area = rice_area = chicken_area = vege_area = tahu_area = tempe_area = 0
        for mask, cls in zip(masks, classes):
            area = mask.sum()
            if cls == 0:
                plate_area += area
            elif cls == 1:
                rice_area += area
            elif cls == 2:
                chicken_area += area
            elif cls == 3:
                vege_area += area
            elif cls == 4:
                tahu_area += area
            elif cls == 5:
                tempe_area += area

        if plate_area > 0:
            data = [
                {"Class": "Rice", "Area %": f"{100*(rice_area/plate_area):.2f}%"},
                {"Class": "Chicken", "Area %": f"{100*(chicken_area/plate_area):.2f}%"},
                {"Class": "Vegetable", "Area %": f"{100*(vege_area/plate_area):.2f}%"},
                {"Class": "Tahu", "Area %": f"{100*(tahu_area/plate_area):.2f}%"},
                {"Class": "Tempe", "Area %": f"{100*(tempe_area/plate_area):.2f}%"},
            ]
            df = pd.DataFrame(data)
            st.markdown("---")
            st.subheader("📊 Area Percentage Relative to Plate")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No plate detected — cannot compute area ratios.")
