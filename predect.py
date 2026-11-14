import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PestSense | AI Crop Monitoring",
    page_icon="🌱",
    layout="wide"
)

# --- LOAD PRE-TRAINED MODEL (CACHED) ---
@st.cache_resource
def load_pest_model():
    # Downloads a YOLO11s model pre-trained on 102 agricultural pest species (IP102)
    model_path = hf_hub_download(
        repo_id="underdogquality/yolo11s-pest-detection",
        filename="best.pt"
    )
    return YOLO(model_path)

with st.spinner("Loading PestSense Pre-Trained AI Model..."):
    model = load_pest_model()

# --- HEADER ---
st.title("🌱 PestSense: Intelligent Early Pest Detection & Monitoring")
st.markdown("**Precision Agriculture Dashboard** | *Vision-Based Pest Counting + Environmental Sensor Fusion*")
st.divider()

# --- SIDEBAR: SENSOR DATA & MODEL SETTINGS ---
with st.sidebar:
    st.header("🌡️ Environmental Sensors (DHT11)")
    st.caption("Simulate live field telemetry or connect serial/MQTT stream")
    temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=45.0, value=28.4, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=20, max_value=100, value=72, step=1)

    st.divider()
    st.header("⚙️ AI Detection Parameters")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
    pest_alert_limit = st.number_input("Alert Threshold (Pest Count)", min_value=1, max_value=20, value=3)

# --- MAIN LAYOUT: IMAGE INPUT & RESULTS ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Camera Feed / Leaf Image Input")
    input_mode = st.radio("Select Source:", ["Upload Leaf Image", "Live Camera (ESP32-CAM / Webcam)"], horizontal=True)

    image = None
    if input_mode == "Upload Leaf Image":
        uploaded_file = st.file_uploader("Upload crop or leaf image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        camera_image = st.camera_input("Capture leaf image")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")

    if image is not None:
        st.image(image, caption="Original Input Frame", use_container_width=True)

with col2:
    st.subheader("2. AI Detection & Counting Output")
    if image is not None:
        # Run inference
        results = model.predict(image, conf=conf_threshold, imgsz=640)
        result = results[0]

        # Plot bounding boxes on image (convert BGR to RGB)
        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        st.image(annotated_rgb, caption="PestSense AI Processed Output", use_container_width=True)

        # Extract detection details
        boxes = result.boxes
        pest_count = len(boxes)
        detections_list = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = result.names[cls_id]
            conf = float(box.conf[0].item())
            detections_list.append({
                "Pest Species": cls_name.title(),
                "Confidence (%)": round(conf * 100, 2)
            })
    else:
        st.info("👈 Upload a leaf image or capture a photo to run real-time pest detection.")
        pest_count = 0
        detections_list = []

# --- DASHBOARD METRICS & DATA FUSION ---
st.divider()
st.subheader("3. Data Fusion, Telemetry & Alert System")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Detected Pest Count", f"{pest_count} Pests", delta="Action Required" if pest_count >= pest_alert_limit else "Safe", delta_color="inverse")
m2.metric("Temperature", f"{temperature} °C", "Optimal" if 20 <= temperature <= 32 else "Stress Risk")
m3.metric("Humidity", f"{humidity} %", "High Fungal/Pest Risk" if humidity > 75 else "Normal")

# Determine Crop Status via Vision + Sensor Fusion
if pest_count >= pest_alert_limit:
    status = "CRITICAL ALERT"
elif pest_count > 0:
    status = "WARNING"
else:
    status = "HEALTHY"

m4.metric("System Status", status)

# Alert Banner
if image is not None:
    if pest_count >= pest_alert_limit:
        st.error(
            f"🚨 **ALERT TRIGGERED:** High pest density detected ({pest_count} pests found). "
            f"Current climate ({temperature}°C, {humidity}% RH) favors rapid reproduction. "
            "Recommend targeted bio-pesticide intervention."
        )
    elif pest_count > 0:
        st.warning(
            f"⚠️ **EARLY WARNING:** {pest_count} pest(s) detected on crop foliage. "
            "Continue automated monitoring before infestation spreads."
        )
    else:
        st.success(
            f"✅ **HEALTHY LEAF:** No pests detected above {int(conf_threshold*100)}% confidence. "
            f"Environmental conditions ({temperature}°C, {humidity}% humidity) are stable."
        )

    # Breakdown Table
    if detections_list:
        st.markdown("#### Detailed Pest Breakdown")
        df = pd.DataFrame(detections_list)
        summary_df = df.groupby("Pest Species").agg(
            Count=("Pest Species", "count"),
            Avg_Confidence=("Confidence (%)", "mean")
        ).reset_index()
        summary_df["Avg_Confidence"] = summary_df["Avg_Confidence"].round(2)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
