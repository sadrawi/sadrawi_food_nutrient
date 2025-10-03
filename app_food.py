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
        results = model.predict(img_np, conf=0.5)[0]

        # Visualize the mask overlay
        seg_img = results.plot()  # returns a numpy array with the segmentation mask overlaid

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

        rows = []
        for box in results[0].boxes:
            cls = int(box.cls[0])              # class index
            label = model.names[cls]           # class name
            conf = float(box.conf[0])          # confidence
            rows.append({"label": label, "confidence": round(conf, 3)})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
        # data = {
        # 'Rice': rice_area,
        # 'Chicken': chicken_area,
        # 'Vegetable': vege_area,
        # 'Tahu': tahu_area,
        # 'Tempe': tempe_area,
        # }
        

        # Create DataFrame
        # df = pd.DataFrame(data)

        # st.dataframe(s_df, use_container_width=True)
        # st.write(df)
        # ALLplate = plate_area + rice_area + chicken_area + vege_area + tahu_area + tempe_area
        # print(f"Rice area: {(rice_area/plate_area)*100:5.2f} %")
        # print(f'Chicken area: {(chicken_area/plate_area)*100:5.2f} %')
        # print(f'Vege area: {(vege_area/plate_area)*100:5.2f} %')
        # print(f'Tahu area: {(tahu_area/plate_area)*100:5.2f} %')
        # print(f'Tempe area: {(tempe_area/plate_area)*100:5.2f} %')
        # st.write(ALLplate)
    # # Convert image for YOLO
    # img_array = np.array(image)
    # # if st.button("Run Segmentation"):
    # # Run YOLO segmentation
    # results = model.predict(img_array,conf=0.5)

    # # Get annotated image
    # annotated = results[0].plot()  # numpy array (BGR)

    # # Show output
    # st.image(annotated, caption="Segmented Result", use_container_width=True)
