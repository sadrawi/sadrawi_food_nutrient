import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# Load YOLO segmentation model
model = YOLO("best_food.pt")  # replace with your trained model

st.title("YOLO Segmentation Demo 🖼️")
st.write("Upload an image for YOLO segmentation")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Run Segmentation"):
        # Convert to numpy array
        img_np = np.array(image)

        # Run prediction
        results = model.predict(img_np, conf=0.5)

        # Visualize the mask overlay
        seg_img = results[0].plot()  # returns a numpy array with the segmentation mask overlaid

        # Show result
        st.image(seg_img, 
            caption="Segmentation Result", 
            use_container_width=True)
        
        r = results[0]

        # Original image
        orig_img = r.orig_img.copy()

        # Get masks and classes
        masks = r.masks.data.cpu().numpy()   # (N, H, W)
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # Compute areas
        plate_area, rice_area, chicken_area, vege_area, tahu_area, tempe_area  = 0, 0, 0, 0, 0, 0
        for mask, cls in zip(masks, classes):
            area = mask.sum()
            if cls == 0:
                plate_area+= area
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

        data = [
            {"class": "rice", "area": rice_area/plate_area},
            {"class": "chicken", "area": chicken_area/plate_area},
            {"class": "vegetable", "area": vege_area/plate_area},
            {"class": "tahu", "area": tahu_area/plate_area},
            {"class": "tempe", "area": tempe_area/plate_area},
        ]
        df = pd.DataFrame(data)
        st.dataframe(df)