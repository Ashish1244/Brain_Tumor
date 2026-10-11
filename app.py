import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Diagnostic Portal",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (CSS Injection)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 16px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value-healthy {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10b981;
    }
    .metric-value-warning {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f43f5e;
    }
    .metric-sub {
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-top: 4px;
    }
    /* Banner Styling */
    .banner-container {
        padding: 20px;
        border-radius: 12px;
        background: radial-gradient(circle at top left, #1e3a8a, #0f172a);
        border: 1px solid #1e40af;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants & Model Setup
# ---------------------------------------------------------
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_DESCRIPTIONS = {
    'Glioma': 'Tumor originating in the glial cells of the brain or spine, requiring rapid oncological staging.',
    'Meningioma': 'Typically benign tumor arising from the meningeal membranes surrounding the brain.',
    'No Tumor': 'Scan exhibits standard tissue morphology with no noticeable focal masses.',
    'Pituitary': 'Abnormal growth in the pituitary gland affecting critical endocrine regulatory functions.'
}

@st.cache_resource(show_spinner=False)
def load_model_cached():
    model_path = 'best_model_finetuned.keras'
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'preprocess_input': preprocess_input}
        )
    except Exception:
        model = tf.keras.models.load_model(model_path)
    return model

with st.spinner("Initializing Deep Learning Engine..."):
    model = load_model_cached()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.title("NeuroScan AI")
    st.caption("ResNet50 Transfer Learning System")
    
    st.markdown("---")
    st.markdown("### ⚙️ Diagnostic Config")
    threshold = st.slider("Confidence Warning Threshold", min_value=50, max_value=95, value=75, step=5)
    st.info("Predictions below this threshold flag a requirement for mandatory senior radiologist review.")
    
    st.markdown("---")
    st.markdown("### 📋 Supported Target Classes")
    for cls in CLASS_NAMES:
        st.markdown(f"- **{cls}**")

# ---------------------------------------------------------
# Header & Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="banner-container">
    <h1 style="color: #f8fafc; margin-bottom: 6px;">🧠 Automated Brain Tumor Detection</h1>
    <p style="color: #94a3b8; font-size: 1.05rem; margin-bottom: 0;">
        Clinical decision-support tool analyzing axial/sagittal MRI scans across 4 pathology classifications.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Layout
# ---------------------------------------------------------
col_left, col_right = st.columns([1.1, 1.3], gap="large")

with col_left:
    st.subheader("1. Patient MRI Upload")
    uploaded_file = st.file_uploader(
        "Upload high-resolution MRI scan (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
        help="Upload axial T1-weighted, T2-weighted, or FLAIR scans"
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Current Diagnostic Subject", use_container_width=True)
    else:
        st.info("Awaiting scan input. Please choose a valid image file above.")

with col_right:
    st.subheader("2. AI Diagnostic Assessment")
    
    if uploaded_file is not None:
        # Preprocessing image
        img_rgb = input_image.convert("RGB").resize((224, 224))
        img_array = np.array(img_rgb, dtype=np.float32)
        img_tensor = np.expand_dims(img_array, axis=0)
        
        # Inference
        predictions = model.predict(img_tensor, verbose=0)[0]
        top_idx = int(np.argmax(predictions))
        predicted_label = CLASS_NAMES[top_idx]
        confidence = float(predictions[top_idx]) * 100
        is_healthy = (predicted_label == "No Tumor")

        # Top Diagnostic Card
        css_class = "metric-value-healthy" if is_healthy else "metric-value-warning"
        status_text = "No Focal Lesion Detected" if is_healthy else "Pathology Identified"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Primary Diagnosis</div>
            <div class="{css_class}">{predicted_label} ({confidence:.2f}%)</div>
            <div class="metric-sub">{status_text} • {CLASS_DESCRIPTIONS[predicted_label]}</div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < threshold:
            st.warning(f"⚠️ Confidence level ({confidence:.1f}%) is below the configured threshold ({threshold}%). Manual clinical review recommended.")

        # Class Probability Breakdown
        st.markdown("#### Probability Spectrum")
        for cls_name, prob in zip(CLASS_NAMES, predictions):
            percentage = prob * 100
            st.markdown(f"**{cls_name}** — `{percentage:.2f}%`")
            st.progress(float(prob))
            
    else:
        st.markdown("""
        > Upload an MRI scan on the left to generate inference analysis, prediction metrics, and categorical probability distributions.
        """)

# ---------------------------------------------------------
# Clinical Reference Accordion
# ---------------------------------------------------------
st.markdown("---")
with st.expander("📖 Medical Diagnostics & Class Knowledge Base"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Glioma**")
        st.caption("Infiltrative intra-axial neoplasms showing variable contrast enhancement patterns.")
    with c2:
        st.markdown("**Meningioma**")
        st.caption("Dural-based extra-axial masses displaying sharp margins and uniform enhancement.")
    with c3:
        st.markdown("**No Tumor**")
        st.caption("Normal anatomical parenchymal structures without abnormal focal mass effect.")
    with c4:
        st.markdown("**Pituitary**")
        st.caption("Sellar and suprasellar masses with potential compression of the optic chiasm.")