import cv2
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
import tempfile

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import requests
import os


st.set_page_config(
    page_title="i3L AI-Based Food Segmentation System",
    layout="wide",
    initial_sidebar_state="auto"
)


col1, col2 = st.columns([1, 1])

# with col1:
#     st.markdown(
#         """
#         <div style="
#             display: flex;
#             align-items: center;          /* vertical centering */
#             justify-content: space-between;
#             margin-bottom: 10px;
#         ">
#             <h2 style="
#                 margin: 0;
#                 text-align: center;
#                 flex: 1;
#                 color: #1f77b4;
#                 font-size: 24px;
#                 font-weight: 600;
#             ">
#                 Food Segmentation System
#             </h2>

#             <img src="i3L_SHL.png" width="120">
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


with col1:
    st.markdown("<div style='text-align: rigth;'>", unsafe_allow_html=True)
    st.image("i3L_SHL.png", width=500)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
    """
    <h2 style="
        text-align: center;
        font-size: 64px;
        color: #1f77b4;
    ">
        Food Segmentation System
    </h2>
    """,
    unsafe_allow_html=True
    )


model_path = "best_food.pt"

# Download model if not available
if not os.path.exists(model_path):
    url = "https://huggingface.co/Sadrawi/food_01/resolve/main/best_food.pt"
    with open(model_path, 'wb') as f:
        f.write(requests.get(url).content)

# Load YOLO segmentation model
model = YOLO(model_path)

# -------------------------------------------------
# CLASS NAMES & COLORS
# -------------------------------------------------
CLASS_NAMES = {
    0: "Plate",
    1: "Rice",
    2: "Chicken",
    3: "Vegetable",
    4: "Tahu",
    5: "Tempe"
}

COLORS = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (222, 149, 13),
    3: (30, 222, 13),
    4: (222, 13, 215),
    5: (13, 208, 222),
}

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
# conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
# imgsz = st.sidebar.selectbox("Image size", [512, 640, 768], index=0)

conf = 0.2
imgsz = 512

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload food images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
if uploaded_files:

    for file in uploaded_files:

        # st.markdown("---")
        # st.subheader(f"📷 {file.name}")

        # Read image
        image = Image.open(file).convert("RGB")
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # YOLO inference (use temp file)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model.predict(
                tmp.name,
                imgsz=imgsz,
                conf=conf
            )

        r = results[0]
        if r.masks is None:
            st.warning("No segmentation detected.")
            continue

        overlay = img_bgr.copy()
        masks = r.masks.data.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        # -----------------------------
        # DRAW MASKS + LABELS
        # -----------------------------
        for mask, cls in zip(masks, classes):
            mask = mask.astype(np.uint8)
            if mask.sum() < 200:
                continue

            color = COLORS.get(cls, (0, 255, 0))
            colored_mask = np.zeros_like(overlay)
            for c in range(3):
                colored_mask[:, :, c] = mask * color[c]

            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

            # Skip plate label
            if cls == 0:
                continue

            ys, xs = np.where(mask > 0)
            cx, cy = int(xs.mean()), int(ys.mean())
            label = CLASS_NAMES.get(cls, f"class_{cls}")

            # cv2.putText(overlay, label, (cx - 25, cy),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #             (0, 0, 0), 4, cv2.LINE_AA)
            # cv2.putText(overlay, label, (cx - 25, cy),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #             (255, 255, 255), 2, cv2.LINE_AA)

        # -----------------------------
        # AREA COMPUTATION
        # -----------------------------
        areas = {k: 0 for k in CLASS_NAMES.keys()}
        for mask, cls in zip(masks, classes):
            areas[cls] += mask.sum()

        ALL = sum(areas.values())

        # -----------------------------
        # DRAW VERTICAL PERCENTAGE SUMMARY
        # -----------------------------
        labels = ["Rice", "Chicken", "Vegetable", "Tahu", "Tempe"]
        values = [areas[1], areas[2], areas[3], areas[4], areas[5]]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        x_label, x_colon, x_value = 10, 130, 220
        y = 30

        bg = overlay.copy()
        cv2.rectangle(bg, (0, 0), (260, 170), (0, 0, 0), -1)
        overlay = cv2.addWeighted(bg, 0.4, overlay, 0.6, 0)

        for label, val in zip(labels, values):
            pct = val / ALL * 100 if ALL > 0 else 0
            cv2.putText(overlay, label, (x_label, y),
                        font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)
            cv2.putText(overlay, ":", (x_colon, y),
                        font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

            txt = f"{pct:5.1f}%"
            (tw, _), _ = cv2.getTextSize(txt, font, font_scale, thickness)
            cv2.putText(overlay, txt, (x_value - tw, y),
                        font, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)
            y += 30

        # -----------------------------
        # DISPLAY (SIDE BY SIDE)
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.imshow(img_rgb)
            ax.set_title("Raw Image")
            ax.axis("off")
            ax.add_patch(Rectangle((0, 0), 1, 1,
                         transform=ax.transAxes,
                         fill=False, edgecolor="black", linewidth=2))
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax.set_title("Segmented Image")
            ax.axis("off")
            ax.add_patch(Rectangle((0, 0), 1, 1,
                         transform=ax.transAxes,
                         fill=False, edgecolor="black", linewidth=2))
            st.pyplot(fig)
