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
        results = model.predict(img_np, conf=0.25)

        # Visualize the mask overlay
        # seg_img = results.plot()  # returns a numpy array with the segmentation mask overlaid

        # # Show result
        # st.image(seg_img, 
        #     caption="Segmentation Result", 
        #     use_container_width=True)
        
        r = results[0]

        # Original image
        orig_img = r.orig_img.copy()

        # Get masks and classes
        masks = r.masks.data.cpu().numpy()   # (N, H, W)
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # Assign colors: rice = red, plate = blue
        colors = {
        0: (0, 0, 0),   # plate
        1: (255, 255, 255),    # rice
        2: (222, 149, 13),    # chicken
        3: (30, 222, 13),    # vege
        4: (222, 13, 215),    # tahu
        5: (13, 208, 222),    # tempe

    }

        overlay = orig_img.copy()

        for mask, cls in zip(masks, classes):
            mask = mask.astype(np.uint8)
            color = colors.get(cls, (0, 255, 0))  # default green
            # Create colored mask
            colored_mask = np.zeros_like(orig_img, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]
            # Blend with overlay
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)


        boxes = r.boxes.xyxy.cpu().numpy() # (N, 4)
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            color = colors.get(cls, (0, 255, 0))  # default green
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        plt.figure(figsize=(8,8))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        data = [
            {"class": "rice", "area": rice_area/plate_area},
            {"class": "chicken", "area": chicken_area/plate_area},
            {"class": "vegetable", "area": vege_area/plate_area},
            {"class": "tahu", "area": tahu_area/plate_area},
            {"class": "tempe", "area": tempe_area/plate_area},
        ]
        df = pd.DataFrame(data)
        st.dataframe(df)