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
        "title": "Northern Corn Leaf Blight",
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

# Explicit Rule-Based Matcher: (crop_keys, disease_keys, directory_key)
DISEASE_RULES = [
    (["apple"], ["scab"], "apple scab"),
    (["apple"], ["rust"], "apple rust"),
    (["pepper", "bell"], ["spot"], "bell pepper leaf spot"),
    (["corn"], ["gray"], "corn gray leaf spot"),
    (["corn"], ["blight"], "corn leaf blight"),
    (["corn"], ["rust"], "corn rust"),
    (["potato"], ["early"], "potato leaf early blight"),
    (["potato"], ["late"], "potato leaf late blight"),
    (["squash"], ["mildew", "powdery"], "squash powdery mildew"),
    (["tomato"], ["early"], "tomato early blight"),
    (["tomato"], ["septoria"], "tomato septoria leaf spot"),
    (["tomato"], ["bacterial"], "tomato leaf bacterial spot"),
    (["tomato"], ["late"], "tomato leaf late blight"),
    (["tomato"], ["mosaic"], "tomato leaf mosaic virus"),
    (["tomato"], ["yellow"], "tomato leaf yellow virus"),
    (["tomato"], ["mold"], "tomato mold leaf"),
    (["tomato"], ["mite", "spider"], "tomato two-spotted spider mites"),
    (["grape"], ["rot", "black"], "grape leaf black rot"),
]

ALL_DISEASE_TOKENS = {
    "scab", "rust", "spot", "gray", "blight", "early", "late", 
    "mildew", "powdery", "septoria", "bacterial", "mosaic", 
    "yellow", "mold", "mite", "spider", "rot", "black"
}

def lookup_disease_advisory(label_text):
    """Matches detected label with the disease directory using strict token rules."""
    cleaned = label_text.lower().replace("_", " ").replace("-", " ")

    # 1. Match specific crop + disease combinations
    for crop_tokens, disease_tokens, dict_key in DISEASE_RULES:
        has_crop = any(c in cleaned for c in crop_tokens)
        has_disease = any(d in cleaned for d in disease_tokens)
        if has_crop and has_disease:
            return DISEASE_DIRECTORY[dict_key]

    # 2. Check if the leaf is healthy or baseline foliage
    if "healthy" in cleaned or not any(d in cleaned for d in ALL_DISEASE_TOKENS):
        return DISEASE_DIRECTORY["healthy"]

    return DEFAULT_FALLBACK_ADVISORY

# --------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & ANNOTATION DRAWING
# --------------------------------------------------------------------------
def render_leaf_annotation(img, box_coords, leaf_idx, mode, color):
    """Draws resolution-scaled bounding boxes or arrow callouts."""
    x1, y1, x2, y2 = box_coords
    img_h, img_w, _ = img.shape

    scale = max(1.0, max(img_h, img_w) / 1000.0)

    font_scale = 0.85 * scale
    font_thickness = max(2, int(2.2 * scale))
    line_thickness = max(2, int(2.5 * scale))
    pad = int(8 * scale)

    badge_text = str(leaf_idx)
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    badge_r = max(tw, th) // 2 + pad

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if mode == "Bounding Boxes":
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)
        badge_w = tw + (pad * 2)
        badge_h = th + (pad * 2)
        cv2.rectangle(img, (x1, y1), (x1 + badge_w, y1 + badge_h), (255, 255, 255), -1)
        cv2.rectangle(img, (x1, y1), (x1 + badge_w, y1 + badge_h), color, line_thickness)
        cv2.putText(
            img, badge_text, (x1 + pad, y1 + th + pad - int(2 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
        )
    else:
        offset_dist = int(55 * scale)
        offset_x = -offset_dist if cx > offset_dist + badge_r else offset_dist
        offset_y = -offset_dist if cy > offset_dist + badge_r else offset_dist
        bx = max(badge_r + 5, min(img_w - badge_r - 5, cx + offset_x))
        by = max(badge_r + 5, min(img_h - badge_r - 5, cy + offset_y))

        cv2.circle(img, (cx, cy), max(4, int(5 * scale)), color, -1)
        cv2.circle(img, (cx, cy), max(6, int(7 * scale)), (255, 255, 255), max(1, int(1.5 * scale)))
        cv2.arrowedLine(img, (bx, by), (cx, cy), color, line_thickness, tipLength=0.25)
        cv2.circle(img, (bx, by), badge_r, (255, 255, 255), -1)
        cv2.circle(img, (bx, by), badge_r, color, line_thickness)
        cv2.putText(
            img, badge_text, (bx - tw // 2, by + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
        )


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
# 4. SIDEBAR (CONTROLS & HYPERPARAMETERS)
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Display & Hyperparameters")
    
    view_mode = st.radio(
        "Leaf Annotation Style",
        options=["Pointing Arrows", "Bounding Boxes"],
        index=0,
        help="Pointing arrows leave leaf symptoms clear; bounding boxes outline the full detected region."
    )
    
    st.divider()
    st.subheader("Inference Settings")
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=1.00,
        value=0.25,
        step=0.01,
        help="Matches Colab default (0.25). Higher values eliminate weak, noisy candidate boxes."
    )
    
    iou_threshold = st.slider(
        "IoU (NMS) Overlap Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Lower values merge coinciding duplicate boxes on the same leaf."
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
# 6. INFERENCE & RESULTS PROCESSING
# --------------------------------------------------------------------------
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    orig_rgb = np.array(pil_image)
    
    is_unrestricted = (selected_option == DEFAULT_ALL_OPTION)
    
    if is_unrestricted:
        allowed_class_ids = None
        user_specified_crop = "plant"
    else:
        user_specified_crop = selected_option.lower()
        allowed_class_ids = [
            c_id for c_id, name in model.names.items()
            if user_specified_crop in name.lower().replace("_", " ")
        ]

    with st.spinner("Analyzing leaf condition..."):
        # Pass 1: Unbiased Scan Across All Crops
        results_unconstrained = model.predict(
            source=pil_image,
            conf=conf_threshold,
            iou=iou_threshold,
            agnostic_nms=True,
            save=False,
            verbose=False
        )

        # Pass 2: Direct Crop-Restricted Diagnosis
        results_targeted = model.predict(
            source=pil_image,
            classes=allowed_class_ids,
            conf=conf_threshold,
            iou=iou_threshold,
            agnostic_nms=True,
            save=False,
            verbose=False
        )

        img_raw = orig_rgb.copy()
        img_filtered = orig_rgb.copy()

        unconstrained_boxes = results_unconstrained[0].boxes
        targeted_boxes = results_targeted[0].boxes

        # Check for domain mismatch across all Pass 1 detections
        mismatch_detected = False
        mismatched_crop_name = ""

        if not is_unrestricted and len(unconstrained_boxes) > 0:
            pass1_crops = set()
            for b in unconstrained_boxes:
                c_name = model.names[int(b.cls[0])].lower().replace("___", " ").replace("_", " ")
                for kw in crop_keywords:
                    clean_kw = kw.replace("_", " ")
                    if clean_kw in c_name:
                        pass1_crops.add(clean_kw.title())

            selected_clean = user_specified_crop.replace("_", " ").title()
            other_crops = [c for c in pass1_crops if c != selected_clean]

            # Trigger warning if another crop was identified and (selected crop is absent OR Pass 2 found 0 leaves)
            if other_crops and (selected_clean not in pass1_crops or len(targeted_boxes) == 0):
                mismatch_detected = True
                mismatched_crop_name = ", ".join(sorted(other_crops))

        # Pass 1 Summary
        raw_summary = []
        for idx, box in enumerate(unconstrained_boxes, start=1):
            u_xyxy = box.xyxy[0].cpu().numpy().astype(int)
            u_class_id = int(box.cls[0])
            u_confidence = float(box.conf[0]) * 100
            u_raw_label = model.names[u_class_id].replace("___", " ").replace("_", " ")

            render_leaf_annotation(img_raw, u_xyxy, idx, view_mode, (0, 102, 255))
            raw_summary.append({
                "Leaf #": idx,
                "Identified Condition": u_raw_label.title(),
                "Confidence": f"{u_confidence:.1f}%"
            })

        # Pass 2 Summary
        filtered_summary = []
        detected_conditions = set()
        for idx, box in enumerate(targeted_boxes, start=1):
            t_xyxy = box.xyxy[0].cpu().numpy().astype(int)
            t_class_id = int(box.cls[0])
            t_confidence = float(box.conf[0]) * 100
            t_label = model.names[t_class_id].replace("___", " ").replace("_", " ").title()

            detected_conditions.add(t_label)
            render_leaf_annotation(img_filtered, t_xyxy, idx, view_mode, (0, 180, 80))
            filtered_summary.append({
                "Leaf #": idx,
                "Diagnostic Label": t_label,
                "Confidence": f"{t_confidence:.1f}%"
            })

    # --------------------------------------------------------------------------
    # 7. VISUAL & TABULAR RESULTS
    # --------------------------------------------------------------------------
    st.divider()

    if mismatch_detected:
        match_count = len(filtered_summary)
        suffix = "s" if match_count != 1 else ""
        st.warning(
            f"⚠️ **Possible Crop Mismatch:** The open AI scan identified symptoms characteristic of **{mismatched_crop_name} foliage**, "
            f"while diagnosis is strictly restricted to **{selected_option}** (which returned {match_count} match{suffix}). "
            f"Please verify your crop selection in Step 1 if these results look unexpected."
        )

    st.subheader(f"Diagnostic Visualizations ({view_mode})")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pass 1: Unbiased AI Scan (All Crops)**")
        st.image(img_raw, use_container_width=True)

    with col2:
        right_title = (
            "**Pass 2: General Diagnosis Output**"
            if is_unrestricted
            else f"**Pass 2: Direct Restricted Diagnosis ({selected_option})**"
        )
        st.markdown(right_title)
        st.image(img_filtered, use_container_width=True)

    st.markdown("### Diagnostic Summary Table")
    rep_col1, rep_col2 = st.columns(2)
    with rep_col1:
        st.markdown("**Unbiased AI Scan Findings**")
        if raw_summary:
            st.dataframe(raw_summary, use_container_width=True)
        else:
            st.info("No leaf structures detected above the confidence threshold.")

    with rep_col2:
        st.markdown(f"**Targeted Diagnosis ({'All Crops' if is_unrestricted else selected_option})**")
        if filtered_summary:
            st.dataframe(filtered_summary, use_container_width=True)
        else:
            st.info(f"No specific lesions detected within the {selected_option} domain.")

    # --------------------------------------------------------------------------
    # 8. PATHOLOGY ADVISORY (DEDUPLICATED BY ADVISORY TITLE)
    # --------------------------------------------------------------------------
    advisory_targets = detected_conditions if detected_conditions else {r["Identified Condition"] for r in raw_summary}

    if advisory_targets:
        st.divider()
        st.subheader("Pathology Advisory & Remediation Directives")

        # Map each label to its info profile and deduplicate by title
        unique_advisories = {}
        for condition in advisory_targets:
            info = lookup_disease_advisory(condition)
            unique_advisories[info["title"]] = info

        # Render one card per unique disease
        for title, info in sorted(unique_advisories.items()):
            with st.expander(f"Treatment Profile: {title}", expanded=True):
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
# 9. AGRONOMIC DISCLAIMER (DARK LOWER SECTION)
# --------------------------------------------------------------------------
st.divider()

footer_html = """
<div style="background-color: #111827; color: #e5e7eb; padding: 24px 28px; border-radius: 12px; margin-top: 20px; border: 1px solid #374151; font-family: inherit;">
<div style="background-color: #1f2937; border-left: 4px solid #f59e0b; padding: 16px 20px; border-radius: 6px; color: #d1d5db; font-size: 0.95rem; line-height: 1.5;">
<strong style="color: #f59e0b; font-size: 1rem;">Agronomic Disclaimer:</strong><br>
This AI diagnostic system is designed strictly for research, educational, and assistive decision-support purposes. While the model utilizes advanced vision pipelines, environmental factors, lighting, and co-infections can influence accuracy. Recommendations should be cross-referenced with regional agricultural extension services or certified crop agronomists before administering chemical treatments.
</div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
