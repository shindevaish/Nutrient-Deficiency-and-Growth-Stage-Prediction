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
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #1f6f4a 0%, #38a169 100%);
        padding: 28px;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 28px rgba(31, 45, 31, 0.12);
    }

    .hero-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .hero-sub {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin: 0;
    }

    .soft-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid #e3ecdf;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 24px rgba(31, 45, 31, 0.08);
    }

    .result-green {
        background: linear-gradient(135deg, #15803d 0%, #22c55e 100%);
        border-radius: 18px;
        padding: 20px;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,0.12);
    }

    .result-red {
        background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);
        border-radius: 18px;
        padding: 20px;
        color: white;
        box-shadow: 0 10px 24px rgba(0,0,0,0.12);
    }

    .metric-label {
        font-size: 0.95rem;
        opacity: 0.92;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-sub {
        margin-top: 10px;
        font-size: 0.95rem;
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
        box-shadow: 0 8px 18px rgba(47, 133, 90, 0.28);
    }

    .stButton button:hover {
        filter: brightness(1.03);
        transition: 0.2s ease;
    }

    div[data-testid="stFileUploader"] {
        background: #f9fcf8;
        border-radius: 14px;
        padding: 10px;
        border: 1px solid #dbe8d7;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 1.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)

CROP_MAP = {
    "Rice": 0,
    "Maize": 1,
    "Coffee": 2
}

DEF_LABELS = {
    0: "Nitrogen (N)",
    1: "Phosphorus (P)",
    2: "Potassium (K)",
    3: "Healthy"
}

GROWTH_LABELS = {
    0: "Stage 1 (Seedling)",
    1: "Stage 2 (Vegetative)",
    2: "Stage 3 (Flowering)"
}


@st.cache_resource
def load_tflite_interpreters():
    def_interpreter = tf.lite.Interpreter(model_path="deficiency_300px_autosave.tflite")
    gro_interpreter = tf.lite.Interpreter(model_path="best_growth_fusion_model_v2.tflite")

    def_interpreter.allocate_tensors()
    gro_interpreter.allocate_tensors()

    return def_interpreter, gro_interpreter


def preprocess_image(img, size):
    img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def prepare_meta_input(selected_crop):
    meta_input = np.zeros((1, 3), dtype=np.float32)
    meta_input[0, CROP_MAP[selected_crop]] = 1.0
    return meta_input


def set_tflite_inputs(interpreter, image_tensor, meta_tensor):
    input_details = interpreter.get_input_details()

    if len(input_details) == 1:
        input_index = input_details[0]["index"]
        input_dtype = input_details[0]["dtype"]
        interpreter.set_tensor(input_index, image_tensor.astype(input_dtype))
        return

    for inp in input_details:
        input_name = inp["name"].lower()
        input_index = inp["index"]
        input_dtype = inp["dtype"]

        if "crop" in input_name or "meta" in input_name:
            interpreter.set_tensor(input_index, meta_tensor.astype(input_dtype))
        else:
            interpreter.set_tensor(input_index, image_tensor.astype(input_dtype))


def run_tflite_inference(interpreter, image_tensor, meta_tensor):
    set_tflite_inputs(interpreter, image_tensor, meta_tensor)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


def safe_confidence(pred):
    return float(np.max(pred)) * 100.0


def main():
    def_interpreter, gro_interpreter = load_tflite_interpreters()

    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">🌱 Smart Crop Health & Growth Analyzer</div>
        <p class="hero-sub">
            Upload a leaf image, choose the crop type, and get AI-powered deficiency and growth-stage diagnosis in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Upload & Settings")
        selected_crop = st.selectbox("Select Crop Type", list(CROP_MAP.keys()))
        uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
        analyze = st.button("🔍 Analyze Plant")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Image Preview")
        if uploaded_file is not None:
            preview_img = Image.open(uploaded_file).convert("RGB")
            st.image(preview_img, caption="Uploaded Leaf Sample", use_container_width=True)
        else:
            st.info("Upload a JPG or PNG image to preview it here.")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze and uploaded_file is None:
        st.warning("Please upload a leaf image before clicking Analyze Plant.")
        return

    if analyze and uploaded_file is not None:
        try:
            with st.spinner("Analyzing plant health and growth stage..."):
                img = Image.open(uploaded_file).convert("RGB")
                meta_input = prepare_meta_input(selected_crop)

                def_img = preprocess_image(img, (300, 300))
                gro_img = preprocess_image(img, (224, 224))

                def_pred = run_tflite_inference(def_interpreter, def_img, meta_input)
                gro_pred = run_tflite_inference(gro_interpreter, gro_img, meta_input)

                def_idx = int(np.argmax(def_pred))
                gro_idx = int(np.argmax(gro_pred))

                def_res = DEF_LABELS.get(def_idx, "Unknown")
                gro_res = GROWTH_LABELS.get(gro_idx, "Unknown")

                def_conf = safe_confidence(def_pred)
                gro_conf = safe_confidence(gro_pred)

            st.markdown("### Diagnostic Results")
            r1, r2 = st.columns(2, gap="large")

            with r1:
                st.markdown(f"""
                <div class="result-green">
                    <div class="metric-label">Growth Stage</div>
                    <div class="metric-value">{gro_res}</div>
                    <div class="metric-sub">Confidence: {gro_conf:.2f}%</div>
                </div>
                """, unsafe_allow_html=T
