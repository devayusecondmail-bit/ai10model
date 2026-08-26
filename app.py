import os
import urllib.request
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# --------------------------------------------------------------------------
# 1. PAGE SETUP & CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Pathology Diagnostic System",
    layout="wide"
)

MODEL_PATH = "ai10.pt"
MODEL_URL = "https://github.com/devayusecondmail-bit/ai10model/releases/download/v1.0/ai10.pt"

# --------------------------------------------------------------------------
# 2. HELPER FUNCTIONS & MODEL CACHING
# --------------------------------------------------------------------------
def calculate_box_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


@st.cache_resource
def load_yolo_model():
    """Downloads model from release if missing, then caches it in RAM."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)


# Load model
model = load_yolo_model()

# Extract crops from model class names
crop_keywords = [
    "apple", "bell_pepper", "blueberry", "cherry", "corn",
    "grape", "peach", "potato", "raspberry", "soyabean",
    "soybean", "squash", "strawberry", "tomato"
]

available_crops = set()
for class_name in model.names.values():
    clean_name = class_name.lower()
    for kw in crop_keywords:
        if kw in clean_name:
            available_crops.add(kw.replace("_", " ").title())
            break

DEFAULT_ALL_OPTION = "All Crops (General Diagnostic / Unrestricted)"
dropdown_options = [DEFAULT_ALL_OPTION] + sorted(list(available_crops))

# --------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Diagnostic Controls")
    
    selected_option = st.selectbox(
        "Target Crop Category",
        options=dropdown_options,
        index=0,
        help="Select a specific crop to restrict diagnoses to valid pathology profiles."
    )
    
    st.divider()
    st.subheader("Inference Hyperparameters")
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=1.00,
        value=0.10,
        step=0.01,
        help="Lower values detect subtle leaf lesions; higher values reduce false positives."
    )
    
    iou_threshold = st.slider(
        "IoU (NMS) Overlap Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.30,
        step=0.05,
        help="Controls spatial overlap boundary for grouping duplicate bounding boxes."
    )
    
    st.caption("Architecture: YOLOv8 Two-Pass Pathology Pipeline")

# --------------------------------------------------------------------------
# 4. MAIN INTERACTION AREA
# --------------------------------------------------------------------------
st.title("Automated Plant Pathology & Leaf Diagnostic System")
st.write("Upload a leaf image to evaluate potential diseases, verify crop domain integrity, and flag unresolved symptoms.")

uploaded_file = st.file_uploader(
    "Upload Plant Image", 
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    
    is_unrestricted = (selected_option == DEFAULT_ALL_OPTION)
    
    if is_unrestricted:
        allowed_class_ids = list(model.names.keys())
        user_specified_crop = "plant"
    else:
        user_specified_crop = selected_option.lower()
        allowed_class_ids = [
            c_id for c_id, name in model.names.items()
            if user_specified_crop in name.lower().replace("_", " ")
        ]

    with st.spinner("Processing diagnostic passes..."):
        # Pass 1: Raw Unconstrained Inference
        results_unconstrained = model.predict(
            source=orig_rgb,
            conf=conf_threshold,
            iou=iou_threshold,
            agnostic_nms=True,
            save=False,
            verbose=False
        )

        # Pass 2: Scrape Locked to Filter Set
        results_targeted = model.predict(
            source=orig_rgb,
            classes=allowed_class_ids,
            conf=0.001,
            iou=iou_threshold,
            agnostic_nms=True,
            save=False,
            verbose=False
        )

        img_raw = orig_rgb.copy()
        img_filtered = orig_rgb.copy()

        unconstrained_boxes = results_unconstrained[0].boxes
        targeted_boxes = results_targeted[0].boxes

        raw_summary = []
        filtered_summary = []
        leaf_idx = 1

        for box in unconstrained_boxes:
            u_xyxy = box.xyxy[0].cpu().numpy().astype(int)
            ux1, uy1, ux2, uy2 = u_xyxy

            u_class_id = int(box.cls[0])
            u_confidence = float(box.conf[0]) * 100
            u_raw_label = model.names[u_class_id].replace("___", " ").replace("_", " ")

            badge_text = str(leaf_idx)
            font_scale = max(0.6, min(1.0, (ux2 - ux1) / 300))
            font_thickness = 2
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            cv2.rectangle(img_raw, (ux1, uy1), (ux2, uy2), (0, 102, 255), 3)
            cv2.rectangle(img_raw, (ux1, uy1), (ux1 + tw + 12, uy1 + th + 12), (255, 255, 255), -1)
            cv2.rectangle(img_raw, (ux1, uy1), (ux1 + tw + 12, uy1 + th + 12), (0, 102, 255), 2)
            cv2.putText(img_raw, badge_text, (ux1 + 6, uy1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            raw_summary.append({
                "Index": leaf_idx,
                "Identified Label": u_raw_label.title(),
                "Confidence": f"{u_confidence:.1f}%"
            })

            best_target_box = None
            best_target_conf = -1.0

            for t_box in targeted_boxes:
                t_xyxy = t_box.xyxy[0].cpu().numpy().astype(int)
                iou = calculate_box_iou(u_xyxy, t_xyxy)

                if iou > iou_threshold:
                    t_conf = float(t_box.conf[0])
                    if t_conf > best_target_conf:
                        best_target_conf = t_conf
                        best_target_box = t_box

            box_color = (0, 180, 80)
            status_text = "Verified Match"

            if u_class_id in allowed_class_ids:
                final_label = u_raw_label.title()
                final_conf = f"{u_confidence:.1f}%"
            elif best_target_box is not None:
                t_class_id = int(best_target_box.cls[0])
                final_label = model.names[t_class_id].replace("___", " ").replace("_", " ").title()
                final_conf = f"{(best_target_conf * 100):.1f}%"
                status_text = f"Corrected from {u_raw_label.title()}"
            else:
                fallback_crop = user_specified_crop.strip().title()
                final_label = f"{fallback_crop} Leaf (Unresolved Condition)"
                final_conf = "N/A"
                box_color = (230, 130, 0)
                status_text = "Out of Target Domain"

            cv2.rectangle(img_filtered, (ux1, uy1), (ux2, uy2), box_color, 3)
            cv2.rectangle(img_filtered, (ux1, uy1), (ux1 + tw + 12, uy1 + th + 12), (255, 255, 255), -1)
            cv2.rectangle(img_filtered, (ux1, uy1), (ux1 + tw + 12, uy1 + th + 12), box_color, 2)
            cv2.putText(img_filtered, badge_text, (ux1 + 6, uy1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            filtered_summary.append({
                "Index": leaf_idx,
                "Final Diagnostic": final_label,
                "Confidence": final_conf,
                "Status": status_text
            })

            leaf_idx += 1

    # --------------------------------------------------------------------------
    # 5. VISUAL RESULTS DISPLAY
    # --------------------------------------------------------------------------
    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pass 1: Raw Unconstrained Detections")
        st.image(img_raw, use_container_width=True)

    with col2:
        right_title = (
            "Pass 2: Unrestricted Diagnostic"
            if is_unrestricted
            else f"Pass 2: Domain-Restricted ({selected_option})"
        )
        st.subheader(right_title)
        st.image(img_filtered, use_container_width=True)

    # --------------------------------------------------------------------------
    # 6. STRUCTURED DIAGNOSTIC REPORT
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("Diagnostic Summary Report")

    rep_col1, rep_col2 = st.columns(2)

    with rep_col1:
        st.markdown("**Pass 1 Detections**")
        if raw_summary:
            st.dataframe(raw_summary, use_container_width=True)
        else:
            st.info("No leaf structures detected above the confidence threshold.")

    with rep_col2:
        st.markdown("**Pass 2 Verified Diagnosis**")
        if filtered_summary:
            st.dataframe(filtered_summary, use_container_width=True)
        else:
            st.info("No leaf structures verified within diagnostic criteria.")
