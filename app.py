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
        --card: rgba(255, 255, 255, 0.88);
        --card-2: #ffffff;
        --text: #1f2d1f;
        --muted: #667085;
        --line: #d8e2d3;
        --primary: #2f855a;
        --primary-dark: #256b48;
        --healthy: #16a34a;
        --danger: #dc2626;
        --warning: #d97706;
        --shadow: 0 10px 30px rgba(31, 45, 31, 0.10);
        --radius-lg: 20px;
        --radius-md: 14px;
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
        border-radius: var(--radius-lg);
        background: linear-gradient(135deg, #1f6f4a 0%, #38a169 100%);
        color: white;
        box-shadow: var(--shadow);
        margin-bottom: 1.4rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .hero p {
        margin-top: 8px;
        margin-bottom: 0;
        font-size: 1rem;
        color: rgba(255,255,255,0.88);
    }

    .pill-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .pill {
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.20);
        font-size: 0.9rem;
    }

    .card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.55);
        backdrop-filter: blur(8px);
        border-radius: var(--radius-lg);
        padding: 22px;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
        color: var(--text);
    }

    .section-sub {
        font-size: 0.95rem;
        color: var(--muted);
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

    .result-amber {
        background: linear-gradient(135deg, #b45309 0%, #f59e0b 100%);
    }

    .result-label {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .mini-note {
        font-size: 0.92rem;
        color: var(--muted);
        margin-top: 8px;
    }

    .preview-box {
        border: 2px dashed #b7cbb2;
        border-radius: 18px;
        padding: 14px;
        background: #fbfdfb;
    }

    .footer {
        text-align: center;
        color: var(--muted);
        padding: 10px 0 0 0;
        font-size: 0.92rem;
    }

    div[data-testid="stFileUploader"] {
        background: #f9fcf8;
        border-radius: 14px;
        padding: 10px;
        border: 1px solid #dbe8d7;
    }

    div[data-testid="stSelectbox"] > div {
        border-radius: 12px;
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
        box-shadow: 0 8px 18px rgba(47, 133, 90, 0.28);
    }

    .stButton button:hover {
        filter: brightness(1.03);
        transform: translateY(-1px);
        transition: 0.2s ease;
    }

    @media (max-width: 768px) {
        .hero h1 {
            font-size: 1.7rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG & LABELS ----------------
CROP_MAP = {"Rice": 0, "Maize": 1, "Coffee": 2}
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

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_all_models():
    def_mod = tf.keras.models.load_model("deficiency_300px_autosave.keras")
    gro_mod = tf.keras.models.load_model("best_growth_fusion_model_v2.keras")
    return def_mod, gro_mod

def_model, gro_model = load_all_models()

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🌱 Smart Crop Health & Growth Analyzer</h1>
    <p>Upload a leaf image, choose the crop type, and get AI-powered deficiency and growth-stage diagnosis in seconds.</p>
    <div class="pill-row">
        <div class="pill">Leaf Image Analysis</div>
        <div class="pill">NPK Deficiency Detection</div>
        <div class="pill">Growth Stage Prediction</div>
        <div class="pill">Crop-wise Metadata Fusion</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload & Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Select crop type and upload a clear leaf image for accurate prediction.</div>', unsafe_allow_html=True)

    selected_crop = st.selectbox("Select Crop Type", list(CROP_MAP.keys()))
    uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

    analyze = st.button("🔍 Analyze Plant")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Image Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Your uploaded sample will appear here before analysis.</div>', unsafe_allow_html=True)

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="preview-box">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Leaf Sample", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload a JPG or PNG image to preview it here.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- INFERENCE ----------------
if uploaded_file and analyze:
    with st.spinner("Analyzing plant health and growth stage..."):
        img = Image.open(uploaded_file).convert("RGB")

        crop_idx = CROP_MAP[selected_crop]
        meta_input = np.zeros((1, 3))
        meta_input[0, crop_idx] = 1

        img_300 = np.array(img.resize((300, 300))) / 255.0
        img_300 = np.expand_dims(img_300, axis=0)
        def_pred = def_model.predict({"image_input": img_300, "crop_input": meta_input}, verbose=0)
        def_res = DEF_LABELS[np.argmax(def_pred)]
        def_conf = float(np.max(def_pred)) * 100

        img_224 = np.array(img.resize((224, 224))) / 255.0
        img_224 = np.expand_dims(img_224, axis=0)
        gro_pred = gro_model.predict({"image_input": img_224, "crop_input": meta_input}, verbose=0)
        gro_res = GROWTH_LABELS[np.argmax(gro_pred)]
        gro_conf = float(np.max(gro_pred)) * 100

    st.markdown("### Diagnostic Results")

    r1, r2 = st.columns(2, gap="large")

    with r1:
        st.markdown(f"""
        <div class="result-card result-green">
            <div class="result-label">Growth Stage</div>
            <div class="result-value">{gro_res}</div>
            <div style="margin-top:10px; font-size:0.95rem;">Confidence: {gro_conf:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        if "Healthy" in def_res:
            st.markdown(f"""
            <div class="result-card result-green">
                <div class="result-label">Health Status</div>
                <div class="result-value">{def_res}</div>
                <div style="margin-top:10px; font-size:0.95rem;">Confidence: {def_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-red">
                <div class="result-label">Health Status</div>
                <div class="result-value">{def_res} Deficiency</div>
                <div style="margin-top:10px; font-size:0.95rem;">Confidence: {def_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Analysis Summary")
    st.write(f"- **Crop Type:** {selected_crop}")
    st.write(f"- **Detected Growth Stage:** {gro_res}")
    st.write(f"- **Detected Health Condition:** {def_res if def_res == 'Healthy' else def_res + ' Deficiency'}")
    st.write("- **Recommendation:** Use this prediction as a screening tool and verify with agronomy guidance if symptoms persist.")
    st.markdown('</div>', unsafe_allow_html=True)

elif analyze and not uploaded_file:
    st.warning("Please upload a leaf image before clicking Analyze Plant.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Developed with Streamlit, TensorFlow, and image-based crop intelligence.
</div>
""", unsafe_allow_html=True)
