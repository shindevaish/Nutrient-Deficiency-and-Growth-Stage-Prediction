import cv2 # Make sure to add opencv-python to your requirements.txt

# 1. ---------------- NEW: AGRI-ADVICE DICTIONARY ----------------
AGRI_ADVICE = {
    "Nitrogen (N)": {
        "Symptom": "Yellowing starts at leaf tips and moves along the midrib in a V-shape.",
        "Fix": "Apply 50-100kg/ha of Urea. Ensure balanced irrigation to prevent leaching.",
        "Fertilizer": "Urea (46-0-0) or Blood Meal."
    },
    "Phosphorus (P)": {
        "Symptom": "Stunted growth with dark green or purplish/reddish leaf tints.",
        "Fix": "Apply phosphate fertilizers close to the root zone early in the season.",
        "Fertilizer": "DAP (Diammonium Phosphate) or Single Super Phosphate (SSP)."
    },
    "Potassium (K)": {
        "Symptom": "Yellowing or 'scorching' of the leaf edges/margins.",
        "Fix": "Apply Muriate of Potash. Check soil for salinity issues.",
        "Fertilizer": "Potassium Chloride (MOP)."
    },
    "Healthy": {
        "Symptom": "No nutrient deficiencies detected. Leaf shows optimal chlorophyll distribution.",
        "Fix": "Maintain current nutrient management and regular irrigation.",
        "Fertilizer": "Standard balanced NPK (if needed per schedule)."
    }
}

# 2. ---------------- NEW: GRAD-CAM FUNCTION ----------------
def generate_gradcam(model, img_array, meta_input):
    # 'top_activation' is the standard last conv layer name for EfficientNetV2S
    last_conv_layer_name = "top_activation"
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([img_array, meta_input])
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
    return heatmap

# 3. ---------------- UPDATED DEFICIENCY PAGE ----------------
def deficiency_page():
    render_header()
    st.markdown("## 🧪 Deficiency Prediction & Reasoning")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # ... (keep your selectbox and file_uploader code here)
        selected_crop = st.selectbox("Select Crop Type", list(CROP_MAP.keys()), key="def_crop")
        uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], key="def_upload")
        analyze = st.button("Analyze Deficiency", key="def_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file and analyze:
        with st.spinner("Analyzing and generating reasoning..."):
            img = Image.open(uploaded_file).convert("RGB")
            meta_input = prepare_meta_input(selected_crop)
            img_300 = preprocess_image(img, (300, 300))

            # --- Prediction ---
            def_pred = def_model.predict({"image_input": img_300, "crop_input": meta_input}, verbose=0)
            def_idx = int(np.argmax(def_pred))
            def_res = DEF_LABELS[def_idx]
            def_conf = float(np.max(def_pred)) * 100

            # --- Reasoning (Grad-CAM) ---
            heatmap = generate_gradcam(def_model, img_300, meta_input)
            
            # Process heatmap for overlay
            heatmap_resized = cv2.resize(heatmap, (300, 300))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Overlay heatmap on original resized image
            original_img_np = np.uint8(img_300[0] * 255)
            superimposed_img = cv2.addWeighted(original_img_np, 0.6, heatmap_color, 0.4, 0)

        # --- Display Results ---
        res_col, advice_col = st.columns([1, 1])

        with res_col:
            st.markdown(f"""
            <div class="result-card {'result-green' if def_res=='Healthy' else 'result-red'}">
                <div class="result-label">Diagnosis</div>
                <div class="result-value">{def_res}</div>
                <div>Confidence: {def_conf:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.image(superimposed_img, caption="AI Reasoning Heatmap (Red shows key symptoms)", use_container_width=True)

        with advice_col:
            advice = AGRI_ADVICE.get(def_res)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Visual Reason:** {advice['Symptom']}")
            st.markdown(f"**Recommended Action:** {advice['Fix']}")
            st.markdown(f"**Fertilizer:** `{advice['Fertilizer']}`")
            st.markdown('</div>', unsafe_allow_html=True)
