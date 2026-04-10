import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIG & LABELS ---
st.set_page_config(page_title="Agri-Tech Diagnostic", layout="wide")

CROP_MAP = {"Rice": 0, "Maize": 1, "Coffee": 2}
DEF_LABELS = {0: "Nitrogen (N)", 1: "Phosphorus (P)", 2: "Potassium (K)", 3: "Healthy"}
GROWTH_LABELS = {0: "Stage 1 (Seedling)", 1: "Stage 2 (Vegetative)", 2: "Stage 3 (Flowering)"}

# --- MODEL LOADING ---
@st.cache_resource
def load_all_models():
    # Replace with your actual filenames
    def_mod = tf.keras.models.load_model('deficiency_300px_autosave.keras')
    gro_mod = tf.keras.models.load_model('best_growth_fusion_model_v2.keras')
    return def_mod, gro_mod

def_model, gro_model = load_all_models()

# --- UI LAYOUT ---
st.title("🌱 Smart Crop Health & Growth Analyzer")
st.markdown("Upload a leaf image and select the crop type for a full diagnostic.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload & Settings")
    selected_crop = st.selectbox("Select Crop Type", list(CROP_MAP.keys()))
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

with col2:
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Preview", use_container_width=True)

# --- INFERENCE ---
if uploaded_file and st.button("Analyze Plant"):
    with st.spinner("Processing..."):
        # 1. Prepare Metadata (One-Hot [1, 3])
        crop_idx = CROP_MAP[selected_crop]
        meta_input = np.zeros((1, 3))
        meta_input[0, crop_idx] = 1

        # 2. Deficiency Predict (300px)
        img_300 = np.array(img.resize((300, 300))) / 255.0
        img_300 = np.expand_dims(img_300, axis=0)
        def_pred = def_model.predict({"image_input": img_300, "crop_input": meta_input})
        def_res = DEF_LABELS[np.argmax(def_pred)]

        # 3. Growth Predict (224px - using your best 82% model)
        img_224 = np.array(img.resize((224, 224))) / 255.0
        img_224 = np.expand_dims(img_224, axis=0)
        gro_pred = gro_model.predict({"image_input": img_224, "crop_input": meta_input})
        gro_res = GROWTH_LABELS[np.argmax(gro_pred)]

        # --- RESULTS ---
        st.divider()
        res1, res2 = st.columns(2)
        res1.success(f"**Growth Stage:** {gro_res}")
        
        if "Healthy" in def_res:
            res2.success(f"**Health Status:** {def_res}")
        else:
            res2.error(f"**Health Status:** {def_res} Deficiency")
