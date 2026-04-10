import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Agri-Tech Diagnostic",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #eef7ea 0%, #f7faf7 100%);
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #1f6f4a 0%, #38a169 100%);
        padding: 28px;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(31, 45, 31, 0.12);
    }

    .hero-sub {
        color: rgba(255,255,255,0.88);
        font-size: 1rem;
        margin-bottom: 0;
    }

    div[data-testid="stVerticalBlock"] div:has(> div[data-testid="stFileUploader"]) {
        background: rgba(255,255,255,0.88);
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(31, 45, 31, 0.08);
        border: 1px solid #e5efe1;
    }

    div[data-testid="stVerticalBlock"] div:has(> div[data-testid="stImage"]) {
        background: rgba(255,255,255,0.88);
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(31, 45, 31, 0.08);
        border: 1px solid #e5efe1;
    }

    .metric-box-green {
        background: linear-gradient(135deg, #15803d 0%, #22c55e 100%);
        padding: 20px;
        border-radius: 18px;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,0.12);
    }

    .metric-box-red {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        padding: 20px;
        border-radius: 18px;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,0.12);
    }

    .metric-label {
        font-size: 0.95rem;
        opacity: 0.9;
    }

    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 8px;
    }

    .stButton button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(135deg, #2f855a 0%, #256b48 100%);
        color: white;
        border: none;
        padding: 0.85rem 1rem;
        font-size: 1rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

CROP_MAP = {"Rice": 0, "Maize": 1, "Coffee": 2}
DEF_LABELS = {0: "Nitrogen (N)", 1: "Phosphorus (P)", 2: "Potassium (K)", 3: "Healthy"}
GROWTH_LABELS = {0: "Stage 1 (Seedling)", 1: "Stage 2 (Vegetative)", 2: "Stage 3 (Flowering)"}

@st.cache_resource
def load_tflite_interpreters():
    def_interpreter = tf.lite.Interpreter(model_path="deficiency_300px_autosave.tflite")
    gro_interpreter = tf.lite.Interpreter(model_path="best_growth_fusion_model_v2.tflite")
    def_interpreter.allocate_tensors()
    gro_interpreter.allocate_tensors()
    return def_interpreter, gro_interpreter

def preprocess_image(img, size):
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def prepare_meta_input(selected_crop):
    meta = np.zeros((1, 3), dtype=np.float32)
    meta[0, CROP_MAP[selected_crop]] = 1.0
    return meta

def run_tflite_inference(interpreter, image_tensor, meta_tensor):
    input_details = interpreter.get_input_details()

    for inp in input_details:
        name = inp["name"].lower()
        idx = inp["index"]
        dtype = inp["dtype"]

        if "crop" in name or "meta" in name:
            interpreter.set_tensor(idx, meta_tensor.astype(dtype))
        else:
            interpreter.set_tensor(idx, image_tensor.astype(dtype))

    interpreter.invoke()
    output_details = interpreter.get_output_details()
    return interpreter.get_tensor(output_details[0]["index"])

def_interpreter, gro_interpreter = load_tflite_interpreters()

st.markdown("""
<div class="hero-box">
    <div class="main-title">🌱 Smart Crop Health & Growth Analyzer</div>
    <p class="hero-sub">Upload a leaf image, choose the crop type, and get AI-powered deficiency and growth-stage diagnosis in seconds.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Upload & Settings")
    selected_crop = st.selectbox("Select Crop Type", list(CROP_MAP.keys()))
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    analyze = st.button("Analyze Plant")

with col2:
    st.subheader("Image Preview")
    if uploaded_file is not None:
        preview_img = Image.open(uploaded_file).convert("RGB")
        st.image(preview_img, caption="Uploaded Leaf Sample", use_container_width=True)
    else:
        st.info("Upload a JPG or PNG image to preview it here.")

if uploaded_file is not None and analyze:
    with st.spinner("Analyzing plant health and growth stage..."):
        img = Image.open(uploaded_file).convert("RGB")
        meta_input = prepare_meta_input(selected_crop)

        def_pred = run_tflite_inference(def_interpreter, preprocess_image(img, (300, 300)), meta_input)
        gro_pred = run_tflite_inference(gro_interpreter, preprocess_image(img, (224, 224)), meta_input)

        def_res = DEF_LABELS[int(np.argmax(def_pred))]
        gro_res = GROWTH_LABELS[int(np.argmax(gro_pred))]
        def_conf = float(np.max(def_pred)) * 100
        gro_conf = float(np.max(gro_pred)) * 100

    st.markdown("### Diagnostic Results")
    r1, r2 = st.columns(2, gap="large")

    with r1:
        st.markdown(f"""
        <div class="metric-box-green">
            <div class="metric-label">Growth Stage</div>
            <div class="metric-value">{gro_res}</div>
            <div style="margin-top:10px;">Confidence: {gro_conf:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        css_class = "metric-box-green" if def_res == "Healthy" else "metric-box-red"
        label = def_res if def_res == "Healthy" else f"{def_res} Deficiency"
        st.markdown(f"""
        <div class="{css_class}">
            <div class="metric-label">Health Status</div>
            <div class="metric-value">{label}</div>
            <div style="margin-top:10px;">Confidence: {def_conf:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

elif analyze and uploaded_file is None:
    st.warning("Please upload a leaf image before clicking Analyze Plant.")

st.markdown("---")
st.caption("Developed for Agri-Tech crop diagnosis")
