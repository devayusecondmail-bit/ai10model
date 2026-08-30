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
    page_title="Ai10model - Plant Diagnostic System",
    layout="wide"
)

MODEL_PATH = "ai10.pt"
MODEL_URL = "https://github.com/devayusecondmail-bit/ai10model/releases/download/v1.0/ai10.pt"

# --------------------------------------------------------------------------
# 2. PATHOLOGY KNOWLEDGE BASE & ADVISORY DIRECTORY
# --------------------------------------------------------------------------
DISEASE_DIRECTORY = {
    "apple scab": {
        "title": "Apple Scab Leaf",
        "identify": "Look for velvety, olive-green to dark brown spots on the upper leaf surface. Over time, spots darken into corky black patches, causing leaves to curl, turn yellow, and drop early.",
        "prevent": "Plant scab-resistant varieties, prune branches to let air and sunlight through, and rake up fallen leaves in autumn so spores cannot survive the winter.",
        "cure": "Once a leaf is marked, the spot cannot be erased. Stop the spread to new leaves by spraying copper-based fungicides, sulfur, or myclobutanil starting when buds first open in spring."
    },
    "apple rust": {
        "title": "Apple Rust Leaf (Cedar Apple Rust)",
        "identify": "Look for bright yellow-orange spots on the tops of leaves in spring. Later, tiny orange tubes or spiky cups form directly underneath on the lower leaf surface.",
        "prevent": "Choose rust-resistant apple trees and remove nearby juniper or Eastern red cedar trees (within a 1-mile radius), as the fungus needs both trees to complete its life cycle.",
        "cure": "Apply protective fungicides containing myclobutanil or copper starting when flower buds show pink and continuing until all petals fall."
    },
    "bell pepper leaf spot": {
        "title": "Bell Pepper Leaf Spot (Bacterial Leaf Spot)",
        "identify": "Look for small, water-soaked, yellowish-green spots on leaves that turn brown with yellow edges. Infected leaves quickly turn yellow and drop, leaving fruit vulnerable to sunburn.",
        "prevent": "Use certified disease-free seeds, water only at the soil line (never spray leaves), and rotate crops away from peppers and tomatoes for at least two years.",
        "cure": "Bacteria cannot be eliminated from inside an infected leaf. Spray copper bactericides early to protect healthy leaves, and remove heavily diseased plants."
    },
    "corn gray leaf spot": {
        "title": "Corn Gray Leaf Spot",
        "identify": "Look for long, narrow, rectangular tan or gray spots. The lesions are restricted by leaf veins, giving them distinctly square, straight edges.",
        "prevent": "Choose resistant corn hybrids, till old stalks deep into the soil after harvest to bury fungi, and avoid planting corn in the same field back-to-back.",
        "cure": "Apply foliar fungicides (such as strobilurins or triazoles) before silking if you notice spots moving upward onto the middle leaves."
    },
    "corn leaf blight": {
        "title": "Corn Leaf Blight (Northern Corn Leaf Blight)",
        "identify": "Look for large, long, cigar-shaped grayish-green or tan lesions (1 to 6 inches long) that spread freely across the leaf veins.",
        "prevent": "Plant resistant hybrids, rotate fields away from corn, and plow under leftover crop residue after harvest.",
        "cure": "Spray fungicides (such as azoxystrobin or propiconazole) during early vegetative stages if lesions begin appearing before pollination."
    },
    "corn rust": {
        "title": "Corn Rust Leaf (Common Rust)",
        "identify": "Look for small, raised, cinnamon-brown to reddish powdery blisters (pustules) scattered over both the upper and lower surfaces of the leaves.",
        "prevent": "Choose rust-resistant hybrids and plant early in the season to avoid peak summer spore migrations.",
        "cure": "Apply a foliar fungicide if rust pustules appear heavily on the upper leaves before or during the tasseling stage."
    },
    "potato leaf early blight": {
        "title": "Potato Leaf Early Blight",
        "identify": "Look for dark brown to black spots with concentric rings (resembling a target board) on older, lower leaves, often surrounded by a yellow halo.",
        "prevent": "Mulch the soil to stop soil spores from splashing upward, keep plants well-fertilized (especially with nitrogen), and water the soil directly.",
        "cure": "Pull off infected lower leaves immediately. Spray remaining foliage with copper fungicide, mancozeb, or chlorothalonil to stop spores from spreading."
    },
    "potato leaf late blight": {
        "title": "Potato Leaf Late Blight",
        "identify": "Look for large, irregular, water-soaked dark patches that rapidly turn black and papery. In damp weather, a delicate white fuzzy mold appears on the leaf undersides.",
        "prevent": "Plant only certified disease-free seed potatoes, eliminate volunteer potato plants, and ensure leaves stay as dry as possible.",
        "cure": "Act immediately. Spray healthy parts of the crop with copper or chlorothalonil. If an entire plant is infected, dig it up, bag it, and discard it to protect neighboring crops."
    },
    "squash powdery mildew": {
        "title": "Squash Powdery Mildew Leaf",
        "identify": "Look for patches of white, powdery, flour-like dust spreading across the tops and bottoms of leaves, causing leaves to yellow, dry out, and turn brown.",
        "prevent": "Plant in full sunlight, give plants wide spacing for continuous airflow, and choose resistant squash varieties.",
        "cure": "Spray the leaves thoroughly with neem oil, potassium bicarbonate, or sulfur sprays at the very first sight of white powder."
    },
    "tomato early blight": {
        "title": "Tomato Early Blight Leaf",
        "identify": "Look for brown-to-black spots with concentric target-like rings starting on the bottom leaves. Leaves yellow and die from the ground upward.",
        "prevent": "Trim off lower leaves so foliage doesn't touch the ground, mulch heavily beneath the plants, and water strictly at the base.",
        "cure": "Pinch off infected lower leaves and spray the rest of the plant with copper fungicide or chlorothalonil every 7 to 10 days."
    },
    "tomato septoria leaf spot": {
        "title": "Tomato Septoria Leaf Spot",
        "identify": "Look for many tiny, circular spots with dark brown borders and light gray centers, often peppered with tiny black specks inside the spot.",
        "prevent": "Use drip irrigation to keep leaves dry, apply thick mulch, sanitize garden stakes each season, and rotate tomato beds.",
        "cure": "Strip off affected lower leaves right away and spray healthy foliage with copper-based fungicides to block new infections."
    },
    "tomato leaf bacterial spot": {
        "title": "Tomato Leaf Bacterial Spot",
        "identify": "Look for small, dark, greasy-looking spots that dry into black scabs with yellow halos. The centers often crack and fall out, leaving tiny holes in the leaves.",
        "prevent": "Buy clean seeds and starts, never handle or prune plants when wet, and avoid planting near peppers or eggplants.",
        "cure": "There is no cure once bacteria enter the plant tissue. Pull out severely damaged plants and treat the rest with a copper-mancozeb spray mix to slow transmission."
    },
    "tomato leaf late blight": {
        "title": "Tomato Leaf Late Blight",
        "identify": "Look for large, greasy, dark brown blotches that spread rapidly over leaves and stems. Under cool, wet conditions, a white fungal fuzz develops on leaf undersides.",
        "prevent": "Space plants widely for rapid drying, select late-blight-resistant varieties, and avoid overhead watering.",
        "cure": "Spray copper or chlorothalonil immediately as a shield for uninfected tissue. If stems and large sections turn dark and rot, destroy the entire plant immediately."
    },
    "tomato leaf mosaic virus": {
        "title": "Tomato Leaf Mosaic Virus (ToMV/TMV)",
        "identify": "Look for mottled, patchy patterns of light and dark green on leaves, along with twisting, curled 'fern-like' leaf growth and stunted plants.",
        "prevent": "Wash hands thoroughly with soap before gardening (especially after handling tobacco), sanitize tools, and select resistant seeds labeled 'TMV' or 'ToMV'.",
        "cure": "Incurable. Pull up and discard infected plants in the trash immediately—do not compost them, and do not touch healthy plants without washing first."
    },
    "tomato leaf yellow virus": {
        "title": "Tomato Leaf Yellow Virus (Tomato Yellow Leaf Curl - TYLCV)",
        "identify": "Look for leaves curling strongly upward, cupping, and turning bright yellow around the borders. New growth stays stunted and bunched together like a tight bush.",
        "prevent": "Cover young plants with fine insect netting, lay reflective silver mulch, and use yellow sticky traps to keep whiteflies (the insect carrier) away.",
        "cure": "Incurable. Treat the whitefly population using insecticidal soap or neem oil to stop further spread, and remove infected plants from the garden."
    },
    "tomato mold leaf": {
        "title": "Tomato Mold Leaf (Leaf Mold)",
        "identify": "Look for pale green or yellowish patches on the upper surface of leaves, accompanied by a thick, velvety olive-green to brownish mold directly beneath them.",
        "prevent": "Lower greenhouse or garden humidity, space plants to allow cross-breezes, and avoid getting water on the leaves.",
        "cure": "Thin out dense foliage to allow air to pass through, and treat leaves with copper fungicides or Bacillus subtilis bio-fungicides."
    },
    "tomato two-spotted spider mites": {
        "title": "Tomato Two-Spotted Spider Mites Leaf",
        "identify": "Look for fine yellow or white pinprick speckling (stippling) across the leaves, along with light webbing underneath. Heavily damaged leaves turn bronze, dry out, and die.",
        "prevent": "Keep soil consistently moist and rinse dust off pathways, as spider mites multiply rapidly in hot, dry, dusty settings.",
        "cure": "Thoroughly coat the undersides of leaves with insecticidal soap, neem oil, or horticultural oil, or introduce predatory mites (Phytoseiulus persimilis)."
    },
    "grape leaf black rot": {
        "title": "Grape Leaf Black Rot",
        "identify": "Look for small, round, tan-to-reddish brown spots bordered by a dark ring on leaves, containing tiny black dots. Later, green grapes turn into hard, shriveled, black 'mummies.'",
        "prevent": "Prune grapevines to maximize sunlight and airflow throughout the canopy. Collect and destroy all fallen leaves and dried-up fruit from the previous year.",
        "cure": "Spray vines with myclobutanil, captan, or mancozeb starting early in spring from bud break through several weeks after flowering."
    },
    "healthy": {
        "title": "Healthy Foliage Structure",
        "identify": "The leaf shows clear, uniform coloration without visible signs of pathogen spots, mold, lesions, or insect damage.",
        "prevent": "Continue regular field watering at the soil level, maintain balanced crop nutrition, and scout plants weekly.",
        "cure": "No treatment required. The leaf appears healthy and disease-free."
    }
}

DEFAULT_FALLBACK_ADVISORY = {
    "title": "General Foliar Condition",
    "identify": "Leaf anomalies detected matching general plant pathology markers.",
    "prevent": "Ensure adequate planting distance, sanitize pruning tools between plants, and rotate crops annually.",
    "cure": "Consult regional agricultural extension services for targeted treatment guidelines."
}

def lookup_disease_advisory(label_text):
    """Matches detected label with the disease directory."""
    cleaned = label_text.lower().replace("_", " ").replace("-", " ")
    for key, data in DISEASE_DIRECTORY.items():
        key_tokens = key.split()
        if all(token in cleaned for token in key_tokens):
            return data
    for key, data in DISEASE_DIRECTORY.items():
        if key in cleaned or any(tok in cleaned for tok in key.split() if len(tok) > 3):
            return data
    return DEFAULT_FALLBACK_ADVISORY

# --------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & MODEL CACHING
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
    """Downloads model weights if missing, then caches instance in RAM."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return YOLO(MODEL_PATH)


model = load_yolo_model()

# Extract crop roots for dropdown
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
# 4. SIDEBAR (ADVANCED SETTINGS ONLY)
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Advanced Settings")
    st.caption("Fine-tune AI sensitivity and box overlap parameters.")
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=1.00,
        value=0.10,
        step=0.01,
        help="Lower values catch subtle lesions; higher values show only high-confidence predictions."
    )
    
    iou_threshold = st.slider(
    "IoU (NMS) Overlap Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.70,
    step=0.05,
    help="Higher values allow overlapping neighboring leaves to be detected separately."
)
    )

# --------------------------------------------------------------------------
# 5. MAIN WORKSPACE: HEADER & STEP-BY-STEP WORKFLOW
# --------------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>Ai10model</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.1rem; color: gray;'>"
    "Crop & Disease Identification AI Advisor: Instantly recognize your plants, spot health issues, and get clear treatment advice."
    "</p>", 
    unsafe_allow_html=True
)
st.divider()

# Step 1: Crop Selection
col_crop, _ = st.columns([2, 1])
with col_crop:
    selected_option = st.selectbox(
        "Step 1: Select Target Crop",
        options=dropdown_options,
        index=0,
        help="Choose the crop you are testing to ensure accurate diagnosis."
    )

# Step 2: Image Upload
uploaded_file = st.file_uploader(
    "Step 2: Upload Leaf Photo", 
    type=["jpg", "jpeg", "png", "webp"]
)

# --------------------------------------------------------------------------
# 6. INFERENCE & RESULTS PROCESSING (COLAB-MATCHED)
# --------------------------------------------------------------------------
if uploaded_file is not None:
    # Load image via PIL to match Colab's exact color & dimension handling
    pil_image = Image.open(uploaded_file).convert("RGB")
    orig_rgb = np.array(pil_image)

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

    with st.spinner("Analyzing leaf condition..."):
        # Pass 1: Raw Detection (Matching Colab defaults)
        results_unconstrained = model.predict(
            source=pil_image,
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False
        )

        # Pass 2: Targeted Matching
        results_targeted = model.predict(
            source=pil_image,
            classes=allowed_class_ids,
            conf=0.001,
            iou=iou_threshold,
            save=False,
            verbose=False
        )

        img_raw = orig_rgb.copy()
        img_filtered = orig_rgb.copy()

        unconstrained_boxes = results_unconstrained[0].boxes
        targeted_boxes = results_targeted[0].boxes

        raw_summary = []
        filtered_summary = []
        detected_conditions = set()
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

            # Pass 1 Visuals (Blue)
            cv2.rectangle(img_raw, (ux1, uy1), (ux2, uy2), (0, 102, 255), 3)
            cv2.rectangle(img_raw, (ux1, uy1), (ux1 + tw + 12, uy1 + th + 12), (255, 255, 255), -1)
            cv2.rectangle(img_raw, (ux1, uy1), (ux1 + tw + 12, uy1 + th + 12), (0, 102, 255), 2)
            cv2.putText(img_raw, badge_text, (ux1 + 6, uy1 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            raw_summary.append({
                "Index": leaf_idx,
                "Identified Label": u_raw_label.title(),
                "Confidence": f"{u_confidence:.1f}%"
            })

            # Pass 2 Spatial Matching
            best_target_box = None
            best_target_conf = -1.0

            for t_box in targeted_boxes:
                t_xyxy = t_box.xyxy[0].cpu().numpy().astype(int)
                iou = calculate_box_iou(u_xyxy, t_xyxy)

                if iou > 0.10:  # Allow spatial overlap linking
                    t_conf = float(t_box.conf[0])
                    if t_conf > best_target_conf:
                        best_target_conf = t_conf
                        best_target_box = t_box

            box_color = (0, 180, 80)
            status_text = "Verified Match"

            if u_class_id in allowed_class_ids:
                final_label = u_raw_label.title()
                final_conf = f"{u_confidence:.1f}%"
                detected_conditions.add(final_label)
            elif best_target_box is not None:
                t_class_id = int(best_target_box.cls[0])
                final_label = model.names[t_class_id].replace("___", " ").replace("_", " ").title()
                final_conf = f"{(best_target_conf * 100):.1f}%"
                status_text = f"Corrected from {u_raw_label.title()}"
                detected_conditions.add(final_label)
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
    # 7. VISUAL & TABULAR RESULTS
    # --------------------------------------------------------------------------
    st.divider()
    st.subheader("Diagnostic Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pass 1: Raw Unconstrained Detections**")
        st.image(img_raw, use_container_width=True)

    with col2:
        right_title = (
            "**Pass 2: Unrestricted Output**"
            if is_unrestricted
            else f"**Pass 2: Domain-Restricted ({selected_option})**"
        )
        st.markdown(right_title)
        st.image(img_filtered, use_container_width=True)

    st.markdown("### Diagnostic Summary Table")
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

    # --------------------------------------------------------------------------
    # 8. PATHOLOGY ADVISORY (IDENTIFY, PREVENT, CURE)
    # --------------------------------------------------------------------------
    if detected_conditions:
        st.divider()
        st.subheader("Pathology Advisory & Remediation Directives")
        
        for condition in sorted(list(detected_conditions)):
            info = lookup_disease_advisory(condition)
            
            with st.expander(f"Treatment Profile: {info['title']}", expanded=True):
                col_id, col_prev, col_cure = st.columns(3)
                
                with col_id:
                    st.markdown("**How to Identify**")
                    st.write(info["identify"])
                    
                with col_prev:
                    st.markdown("**How to Prevent**")
                    st.write(info["prevent"])
                    
                with col_cure:
                    st.markdown("**How to Cure**")
                    st.write(info["cure"])

# --------------------------------------------------------------------------
# 9. DISCLAIMER & CREDITS (DARK LOWER SECTION)
# --------------------------------------------------------------------------
st.divider()

footer_html = """
<div style="background-color: #111827; color: #e5e7eb; padding: 28px 32px; border-radius: 12px; margin-top: 20px; border: 1px solid #374151; font-family: inherit;">
<div style="background-color: #1f2937; border-left: 4px solid #f59e0b; padding: 16px 20px; border-radius: 6px; margin-bottom: 24px; color: #d1d5db; font-size: 0.95rem; line-height: 1.5;">
<strong style="color: #f59e0b; font-size: 1rem;">Agronomic Disclaimer:</strong><br>
This AI diagnostic system is designed strictly for research, educational, and assistive decision-support purposes. While the model utilizes advanced vision pipelines, environmental factors, lighting, and co-infections can influence accuracy. Recommendations should be cross-referenced with regional agricultural extension services or certified crop agronomists before administering chemical treatments.
</div>
<h3 style="color: #ffffff; margin-bottom: 12px; font-size: 1.25rem;">Project Credits</h3>
<p style="color: #9ca3af; margin-bottom: 12px; font-size: 0.95rem;">Developed by Class 12 students of DPS East:</p>
<ul style="color: #f3f4f6; line-height: 1.8; font-size: 1rem; margin: 0; padding-left: 20px;">
<li><strong>Atharva Kumar</strong></li>
<li><strong>Anshuman Samal</strong></li>
<li><strong>Krish Mamidala</strong></li>
<li><strong>Devayu Singh Thakur</strong></li>
</ul>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
