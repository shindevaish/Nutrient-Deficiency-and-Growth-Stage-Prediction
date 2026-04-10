import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agri-Tech Diagnostic",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    :root {
        --bg: #f4f8f2;
        --card: rgba(255, 255, 255, 0.9);
        --text: #1f2d1f;
        --muted: #667085;
        --primary: #2f855a;
        --primary-dark: #256b48;
        --healthy: #16a34a;
        --danger: #dc2626;
        --shadow: 0 10px 30px rgba(31, 45, 31, 0.10);
    }

    .stApp {
        background: linear-gradient(180deg, #eef7ea 0%, #f7faf7 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .hero {
        padding: 28px 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #1f6f4a 0%, #38a169 100%);
        color: white;
        box-shadow: var(--shadow);
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
    }

    .hero p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 1rem;
        color: rgba(255,255,255,0.88);
    }

    .card {
        background: var(--card);
        border-radius: 20px;
        padding: 22px;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .result-card {
        border-radius: 18px;
        padding: 20px;
        color: white;
        box-shadow: var(--shadow);
        min-height: 140px;
    }

    .result-green {
        background: linear-gradient(135deg, #15803d 0%, #22c55e 100%);
    }

    .result-red {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
    }

    .stButton button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        padding: 0.85rem 1rem;
        font-size: 1rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG & LABELS ----------------
CROP_MAP = {"Rice": 0, "Maize": 1, "Coffee": 2}
DEF_LABELS = {0: "Nitrogen (N)", 1: "Phosphorus (P)", 2: "Potassium (K)", 3: "Healthy"}
GROWTH_LABELS = {0: "Stage 1 (Seedling)", 1: "Stage 2 (Vegetative)", 2: "Stage 3 (Flowering)"}

# ---------------- TFLITE HELPERS ----------------
@st.cache_resource
def load_tflite_interpreters():
    def_interpreter = tf.lite.Interpreter(model_path="deficiency_300px_autosave.tflite")
    gro_interpreter = tf.lite.Interpreter(model_path="best_growth_fusion_model_v2.tflite")

    def_interpreter.allocate_tensors()
    gro_interpreter.allocate_tensors()

    return def_interpreter, gro_interpreter
def_interpreter, gro_interpreter = load_tflite_interpreters()
st.sidebar.write("Deficiency model inputs:", def_interpreter.get_input_details())
st.sidebar.write("Growth model inputs:", gro_interpreter.get_input_details())


def preprocess_image(img, size):
    img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def prepare_meta_input(selected_crop):
    crop_idx = CROP_MAP[selected_crop]
    meta_input = np.zeros((1, 3), dtype=np.float32)
    meta_input[0, crop_idx] = 1.0
    return meta_input

def set_interpreter_inputs(interpreter, image_tensor, meta_tensor):
    input_details = interpreter.get_input_details()

    if len(input_details) != 2:
        raise ValueError(f"Expected 2 inputs, but found {len(input_details)}.")

    for inp in input_details:
        input_name = inp["name"].lower()
        input_index = inp["index"]
        input_dtype = inp["dtype"]

        if "crop" in input_name or "meta" in input_name:
            tensor = meta_tensor.astype(input_dtype)
        else:
            tensor = image_tensor.astype(input_dtype)

        interpreter.set_tensor(input_index, tensor)

def run_tflite_inference(interpreter, image_tensor, meta_tensor):
    set_interpreter_inputs(interpreter, image_tensor, meta_tensor)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data

# ---------------- LOAD MODELS ----------------
def_interpreter, gro_interpreter = load_tflite_interpreters()

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌱 Smart Crop Health & Growth Analyzer</h1>
    <p>Upload a leaf image, choose the crop type, and get AI-powered deficiency and growth-stage diagnosis in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload & Settings")
    selected_crop = st.selectbox("Select Crop Type", list(CROP_MAP.keys()))
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    analyze = st.button("Analyze Plant")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Image Preview")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Leaf Sample", use_container_width=True)
    else:
        st.info("Upload a JPG or PNG image to preview it here.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INFERENCE ----------------
if uploaded_file is not None and analyze:
    with st.spinner("Analyzing plant health and growth stage..."):
        img = Image.open(uploaded_file).convert("RGB")
        meta_input = prepare_meta_input(selected_crop)

        img_300 = preprocess_image(img, (300, 300))
        def_pred = run_tflite_inference(def_interpreter, img_300, meta_input)
        def_idx = int(np.argmax(def_pred))
        def_res = DEF_LABELS[def_idx]
        def_conf = float(np.max(def_pred)) * 100

        img_224 = preprocess_image(img, (224, 224))
        gro_pred = run_tflite_inference(gro_interpreter, img_224, meta_input)
        gro_idx = int(np.argmax(gro_pred))
        gro_res = GROWTH_LABELS[gro_idx]
        gro_conf = float(np.max(gro_pred)) * 100

    st.markdown("### Diagnostic Results")
    res1, res2 = st.columns(2, gap="large")

    with res1:
        st.markdown(f"""
        <div class="result-card result-green">
            <div style="font-size:0.95rem; opacity:0.9;">Growth Stage</div>
            <div style="font-size:1.55rem; font-weight:800; margin-top:8px;">{gro_res}</div>
            <div style="margin-top:10px; font-size:0.95rem;">Confidence: {gro_conf:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        if def_res == "Healthy":
            st.markdown(f"""
            <div class="result-card result-green">
                <div style="font-size:0.95rem; opacity:0.9;">Health Status</div>
                <div style="font-size:1.55rem; font-weight:800; margin-top:8px;">{def_res}</div>
                <div style="margin-top:10px; font-size:0.95rem;">Confidence: {def_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-red">
                <div style="font-size:0.95rem; opacity:0.9;">Health Status</div>
                <div style="font-size:1.55rem; font-weight:800; margin-top:8px;">{def_res} Deficiency</div>
                <div style="margin-top:10px; font-size:0.95rem;">Confidence: {def_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

elif analyze and uploaded_file is None:
    st.warning("Please upload a leaf image before clicking Analyze Plant.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed for Agri-Tech crop diagnosis</p>",
    unsafe_allow_html=True
)
