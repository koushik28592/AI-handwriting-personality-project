import json
import os
from pathlib import Path
from datetime import datetime
from urllib.request import urlopen

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image

try:
    from .gradcam import make_gradcam, overlay_heatmap, preprocess_image
except ImportError:
    from gradcam import make_gradcam, overlay_heatmap, preprocess_image

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
STYLES_PATH = ROOT / "app" / "styles.css"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
DEFAULT_CLASS_NAMES = ["Extrovert", "Introvert", "Optimistic", "Pessimistic", "Stable_Mindset"]

try:
    CLASS_NAMES = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    CLASS_NAMES = DEFAULT_CLASS_NAMES

st.set_page_config(page_title="Cognitive Signal Lab", page_icon="✦", layout="wide", initial_sidebar_state="expanded")


def load_styles():
    if STYLES_PATH.exists():
        st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def card(content, class_name="glass-card"):
    st.markdown(f'<div class="{class_name}">{content}</div>', unsafe_allow_html=True)


def metric_card(value, label, accent=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-accent">{accent}</div></div>',
        unsafe_allow_html=True,
    )


def cam_image(cam, shape):
    resized = cv2.resize(cam, (shape[1], shape[0]))
    ivory = np.array([255, 255, 227], dtype=np.float32)
    blue_gray = np.array([109, 129, 150], dtype=np.float32)
    return np.uint8(ivory * (1 - resized[..., None]) + blue_gray * resized[..., None])


def model_files():
    local_models = {
        path.stem.replace("handwriting_personality_", ""): path
        for path in MODELS_DIR.glob("handwriting_personality_*.keras")
    }
    if local_models:
        return local_models
    model_url = os.getenv("HANDWRITING_MODEL_URL")
    if not model_url:
        return {}
    cached_model = Path.home() / ".cache" / "cognitive_signal_lab" / "handwriting_personality_resnet50.keras"
    if not cached_model.exists():
        try:
            cached_model.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(model_url, timeout=60) as response, cached_model.open("wb") as destination:
                destination.write(response.read())
        except Exception:
            if cached_model.exists():
                cached_model.unlink()
            return {}
    return {"resnet50": cached_model}


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def set_page(page):
    st.session_state.page = page


def init_state():
    st.session_state.setdefault("page", "Dashboard")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("analysis", None)


def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand-mark"><div class="brand-icon">✦</div><div><div class="brand-name">Cognitive Signal Lab</div><div class="brand-sub">HANDWRITING RESEARCH CONSOLE</div></div></div>', unsafe_allow_html=True)
        pages = {
            "Dashboard": "⌂  Dashboard",
            "Analyze Handwriting": "✍  Analyze Handwriting",
            "AI Explanation": "◉  AI Explanation",
            "Model Performance": "▦  Model Performance",
            "About Project": "ⓘ  About Project",
        }
        current = st.session_state.page
        selected = st.radio("Navigation", list(pages), index=list(pages).index(current), format_func=lambda item: pages[item], label_visibility="collapsed")
        if selected != current:
            st.session_state.page = selected
            st.rerun()
        st.markdown('<div class="footer-note">Experimental research interface<br>Model outputs are dataset-label predictions, not psychological assessments.</div>', unsafe_allow_html=True)


def dashboard():
    st.markdown('<div class="hero"><div class="eyebrow">AI / COGNITIVE MODELING</div><h1>AI Handwriting<br>Personality Analysis</h1><div class="hero-copy">Explore observable handwriting patterns through a transparent research workflow that combines visual deep learning with measurable image features.</div></div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1], gap="medium")
    with col_a:
        if st.button("✍  Analyze a handwriting sample", type="primary", use_container_width=True):
            set_page("Analyze Handwriting")
            st.rerun()
    with col_b:
        if st.button("▦  View model performance", use_container_width=True):
            set_page("Model Performance")
            st.rerun()
    st.markdown('<div class="section-title">System snapshot</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    for col, value, label, accent in zip(cols, ["3,227", "5", "ResNet50", "AI / DL"], ["Training images", "Personality classes", "Primary visual model", "Research stack"], ["dataset inventory", "dataset labels", "transfer learning", "image intelligence"]):
        with col:
            metric_card(value, label, accent)
    st.markdown('<div class="section-title">How the signal is modeled</div>', unsafe_allow_html=True)
    card('<div class="eyebrow">RESEARCH PIPELINE</div><h3>From visual trace to model output</h3><p class="muted">Handwriting is transformed into image representations and measurable visual features. The model learns patterns associated with the supplied dataset labels, while Grad-CAM provides a view into influential image regions.</p><span class="pill">RGB preprocessing</span><span class="pill">Transfer learning</span><span class="pill">OpenCV features</span><span class="pill">Macro F1 evaluation</span>')
    if st.session_state.history:
        st.markdown('<div class="section-title">Current session</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)


def analysis_page():
    st.markdown('<div class="eyebrow">LIVE INFERENCE</div><h1>Analyze Handwriting</h1><p class="section-note">Upload one sample to generate a model-backed prediction and visual explanation.</p>', unsafe_allow_html=True)
    models = model_files()
    if not models:
        st.error("No trained model is available for inference.")
        st.info("Commit models/handwriting_personality_resnet50.keras or configure HANDWRITING_MODEL_URL in Streamlit secrets.")
        return
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
        selected = st.selectbox("Inference model", sorted(models), index=sorted(models).index("resnet50") if "resnet50" in models else 0)
        uploaded = st.file_uploader("Drop a handwriting image here", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="visible")
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Original sample", use_container_width=True)
            if st.button("✦  Analyze Handwriting", type="primary", use_container_width=True):
                try:
                    uploaded.seek(0)
                    original, tensor = preprocess_image(uploaded)
                    with st.status("Running analysis", expanded=True) as status:
                        st.write("Preparing image tensor")
                        model = load_model(str(models[selected]))
                        st.write(f"Running {selected} inference")
                        cam, class_index, probabilities = make_gradcam(model, tensor)
                        st.write("Generating Grad-CAM explanation")
                        heatmap = overlay_heatmap(original, cam)
                        heatmap_only = cam_image(cam, original.shape)
                        status.update(label="Analysis complete", state="complete", expanded=False)
                    result = {"model": selected, "class_index": class_index, "probabilities": probabilities.tolist(), "original": original, "heatmap": heatmap, "heatmap_only": heatmap_only, "filename": uploaded.name}
                    st.session_state.analysis = result
                    st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "file": uploaded.name, "model": selected, "prediction": CLASS_NAMES[class_index], "confidence": round(float(probabilities[class_index]) * 100, 2)})
                except Exception:
                    st.error("This image could not be analyzed. Please try another supported handwriting image.")
    result = st.session_state.analysis
    if result:
        probabilities = np.asarray(result["probabilities"])
        class_index = int(result["class_index"])
        with right:
            st.markdown(f'<div class="result-card"><div class="result-label">Predicted personality</div><div class="result-name">{CLASS_NAMES[class_index].replace("_", " ")}</div><div class="result-confidence">{probabilities[class_index] * 100:.2f}% confidence · {result["model"]}</div></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top 3 predictions</div>', unsafe_allow_html=True)
            for index in np.argsort(probabilities)[::-1][:3]:
                st.write(f"**{CLASS_NAMES[index].replace('_', ' ')}**  ·  {probabilities[index] * 100:.2f}%")
            chart = px.bar(pd.DataFrame({"Class": [name.replace("_", " ") for name in CLASS_NAMES], "Probability": probabilities * 100}), x="Probability", y="Class", orientation="h", text_auto=".1f", color_discrete_sequence=["#6D8196"])
            chart.update_layout(height=320, margin=dict(l=0, r=0, t=8, b=8), paper_bgcolor="#FFFFE3", plot_bgcolor="#FFFFE3", font_color="#4A4A4A", xaxis_title="Confidence (%)", yaxis_title="")
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
            report = json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "image": result["filename"], "model": result["model"], "prediction": CLASS_NAMES[class_index], "confidence": float(probabilities[class_index]), "probabilities": dict(zip(CLASS_NAMES, probabilities.tolist()))}, indent=2)
            st.download_button("↓  Download prediction report", report, file_name="handwriting_prediction.json", mime="application/json", use_container_width=True)


def explanation_page():
    st.markdown('<div class="eyebrow">MODEL INTERPRETABILITY</div><h1>AI Explanation</h1><p class="section-note">Inspect the visual regions that influenced the most recent model output.</p>', unsafe_allow_html=True)
    result = st.session_state.analysis
    if not result:
        card('<div class="eyebrow">NO ACTIVE ANALYSIS</div><h3>Run an analysis first</h3><p class="muted">The explanation workspace will populate after a handwriting sample is analyzed.</p>')
        return
    original, heatmap, heatmap_only = result["original"], result["heatmap"], result["heatmap_only"]
    tab_a, tab_b, tab_c = st.tabs(["Original image", "Grad-CAM heatmap", "Overlay"])
    with tab_a: st.image(original, use_container_width=True)
    with tab_b: st.image(heatmap_only, use_container_width=True)
    with tab_c: st.image(heatmap, use_container_width=True)
    st.warning("Grad-CAM highlights image regions that influenced the model prediction. It is an interpretability aid, not proof of a person's personality.")


def performance_page():
    st.markdown('<div class="eyebrow">EVALUATION LAB</div><h1>Model Performance</h1><p class="section-note">Only generated evaluation artifacts are shown here. No metrics are fabricated.</p>', unsafe_allow_html=True)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        st.info("Train and evaluate a model first to populate this page.")
        return
    comparison = pd.read_csv(comparison_path)
    selected = st.selectbox("Evaluation model", comparison["model"].tolist(), index=int(comparison["model"].tolist().index("resnet50")) if "resnet50" in comparison["model"].tolist() else 0)
    row = comparison[comparison.model == selected].iloc[0]
    cols = st.columns(5, gap="medium")
    for col, label, key in zip(cols, ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"], ["accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"]):
        with col:
            metric_card(f"{row[key] * 100:.1f}%" if pd.notna(row[key]) else "—", label, "held-out test")
    st.markdown('<div class="section-title">Model comparison</div>', unsafe_allow_html=True)
    comparison_display = comparison.copy()
    for column in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        comparison_display[column] = (comparison_display[column] * 100).round(1)
    st.dataframe(comparison_display, use_container_width=True, hide_index=True)
    matrix_path = REPORTS_DIR / f"confusion_matrix_{selected}.png"
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
        if matrix_path.exists():
            st.image(str(matrix_path), use_container_width=True)
        else:
            st.info("Confusion matrix not available for this model.")
    with right:
        st.markdown('<div class="section-title">Training curves</div>', unsafe_allow_html=True)
        accuracy_path, loss_path = REPORTS_DIR / "accuracy.png", REPORTS_DIR / "loss.png"
        if accuracy_path.exists() and loss_path.exists():
            st.image(str(accuracy_path), use_container_width=True)
            st.image(str(loss_path), use_container_width=True)
        else:
            st.info("Training curves are not available. Train the model first.")


def about_page():
    st.markdown('<div class="eyebrow">RESEARCH NOTES</div><h1>About Project</h1><p class="section-note">A transparent interface for an experimental cognitive modeling workflow.</p>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        card('<div class="eyebrow">OBJECTIVE</div><h3>Observable traces, careful claims</h3><p class="muted">The project explores whether image representations and measurable handwriting features can classify the labels available in the supplied dataset.</p><div class="eyebrow">COGNITIVE MODELING CONNECTION</div><p class="muted">Handwriting → observable behavioral patterns → feature representation → AI model → personality-related classification.</p>')
        card('<div class="eyebrow">STACK</div><span class="pill">TensorFlow</span><span class="pill">ResNet50</span><span class="pill">OpenCV</span><span class="pill">Plotly</span><span class="pill">Streamlit</span><span class="pill">Grad-CAM</span>')
    with right:
        card('<div class="eyebrow">DATASET</div><h3>3,227 image inventory</h3><p class="muted">Five dataset-defined classes: Extrovert, Introvert, Optimistic, Pessimistic, and Stable Mindset. Exact duplicates are removed from the evaluation split.</p><div class="eyebrow">LIMITATIONS</div><p class="muted">The dataset labels are not validated psychological traits. Writer identity metadata is unavailable, so generalization to unseen writers is uncertain. Image conditions, label quality, language, and demographic factors may influence results.</p>')
        card('<div class="eyebrow">FUTURE WORK</div><p class="muted">Writer-grouped validation, external testing, calibration, validated psychological labels, confidence intervals, and trained image-plus-feature fusion.</p>')
    st.error("This is an experimental AI classification system, not a psychological diagnosis or definitive personality assessment.")


def main():
    load_styles()
    init_state()
    sidebar()
    page = st.session_state.page
    if page == "Dashboard":
        dashboard()
    elif page == "Analyze Handwriting":
        analysis_page()
    elif page == "AI Explanation":
        explanation_page()
    elif page == "Model Performance":
        performance_page()
    else:
        about_page()


main()
