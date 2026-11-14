import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import mysql.connector
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PestSense | AI Crop & Environment Monitoring",
    page_icon="🌱",
    layout="wide"
)

# --- LOAD PRE-TRAINED MODEL (CACHED) ---
@st.cache_resource
def load_pest_model():
    model_path = hf_hub_download(
        repo_id="underdogquality/yolo11s-pest-detection",
        filename="best.pt"
    )
    return YOLO(model_path)

with st.spinner("Loading PestSense Pre-Trained AI Model..."):
    model = load_pest_model()

# --- IMAGE PRE-PROCESSING HELPER ---
def enhance_leaf_contrast(pil_img):
    img_np = np.array(pil_img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_rgb = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)

# --- DATABASE FETCH FUNCTION ---
def fetch_soil_data(limit=200):
    conn = mysql.connector.connect(
        host="82.180.143.66",
        user="u263681140_students",
        passwd="testStudents@123",
        database="u263681140_students",
        connect_timeout=10
    )
    query = f"""
        SELECT DateTime, SoilMoisture, EnvarmentTemp, EnvarmentHumi
        FROM SoilData
        ORDER BY DateTime DESC
        LIMIT {int(limit)}
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if not df.empty:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        for col in ["SoilMoisture", "EnvarmentTemp", "EnvarmentHumi"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_values("DateTime").reset_index(drop=True)
    return df

# --- HEADER ---
st.title("🌱 PestSense: Intelligent Early Pest Detection & Monitoring")
st.markdown("**Precision Agriculture Dashboard** | *Vision-Based Pest Counting + Live MySQL Environmental Sensor Fusion*")

# --- SIDEBAR: AI PARAMETERS ONLY ---
with st.sidebar:
    st.header("⚙️ AI Detection Parameters")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    pest_alert_limit = st.number_input("Alert Threshold (Pest Count)", min_value=1, max_value=20, value=3)
    use_clahe = st.checkbox("Enhance Green-on-Green Contrast (CLAHE)", value=True)
    st.divider()
    if st.button("🔄 Sync Latest Sensor Data", use_container_width=True):
        st.rerun()

# --- FETCH LIVE DATABASE RECORDS FOR BOTH TABS ---
db_error = None
env_df = pd.DataFrame()
try:
    env_df = fetch_soil_data(limit=250)
except mysql.connector.Error as err:
    db_error = str(err)

# Extract latest live sensor values (fallback to None if DB empty/unreachable)
if not env_df.empty:
    latest_row = env_df.iloc[-1]
    live_temp = latest_row["EnvarmentTemp"]
    live_humi = latest_row["EnvarmentHumi"]
    live_soil = latest_row["SoilMoisture"]
    last_updated = latest_row["DateTime"]
else:
    live_temp = None
    live_humi = None
    live_soil = None
    last_updated = None

# --- NAVIGATION TABS ---
tab1, tab2 = st.tabs(["🔍 Vision Pest Detection", "📊 Live Field Sensors & Environment Analysis"])

# ==============================================================================
# TAB 1: VISION PEST DETECTION & COUNTING
# ==============================================================================
with tab1:
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
            input_for_model = enhance_leaf_contrast(image) if use_clahe else image
            results = model.predict(input_for_model, conf=conf_threshold, iou=0.45, imgsz=1024)
            result = results[0]

            annotated_bgr = result.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]
            st.image(annotated_rgb, caption="PestSense AI Processed Output", use_container_width=True)

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

    st.divider()
    
    # Header + Last Updated Timestamp from DB
    head_col1, head_col2 = st.columns([2, 1])
    with head_col1:
        st.subheader("3. Data Fusion, Live Telemetry & Alert System")
    with head_col2:
        if last_updated is not None and pd.notna(last_updated):
            st.info(f"🕒 **Sensor Last Updated:** `{last_updated.strftime('%Y-%m-%d %H:%M:%S')}`")
        else:
            st.warning("🕒 **Sensor Last Updated:** `Unavailable`")

    if db_error:
        st.error(f"Database sync warning: {db_error}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Detected Pest Count",
        f"{pest_count} Pests",
        delta="Action Required" if pest_count >= pest_alert_limit else "Safe",
        delta_color="inverse"
    )
    m2.metric(
        "Live Temperature",
        f"{live_temp:.1f} °C" if pd.notna(live_temp) else "N/A",
        "Optimal" if pd.notna(live_temp) and 20 <= live_temp <= 32 else ("Stress Risk" if pd.notna(live_temp) else None)
    )
    m3.metric(
        "Live Humidity",
        f"{live_humi:.1f} %" if pd.notna(live_humi) else "N/A",
        "High Fungal/Pest Risk" if pd.notna(live_humi) and live_humi > 75 else ("Normal" if pd.notna(live_humi) else None)
    )

    if pest_count >= pest_alert_limit:
        status = "CRITICAL ALERT"
    elif pest_count > 0:
        status = "WARNING"
    else:
        status = "HEALTHY"

    m4.metric("System Status", status)

    if image is not None:
        temp_str = f"{live_temp:.1f}°C" if pd.notna(live_temp) else "N/A"
        humi_str = f"{live_humi:.1f}% RH" if pd.notna(live_humi) else "N/A"

        if pest_count >= pest_alert_limit:
            st.error(
                f"🚨 **ALERT TRIGGERED:** High pest density detected ({pest_count} pests found). "
                f"Live field climate ({temp_str}, {humi_str}) favors rapid reproduction."
            )
        elif pest_count > 0:
            st.warning(f"⚠️ **EARLY WARNING:** {pest_count} pest(s) detected on crop foliage.")
        else:
            st.success(
                f"✅ **HEALTHY LEAF:** No pests detected above {int(conf_threshold*100)}% confidence. "
                f"Live climate: {temp_str}, {humi_str}."
            )

        if detections_list:
            st.markdown("#### Detailed Pest Breakdown")
            df_det = pd.DataFrame(detections_list)
            summary_df = df_det.groupby("Pest Species").agg(
                Count=("Pest Species", "count"),
                Avg_Confidence=("Confidence (%)", "mean")
            ).reset_index()
            summary_df["Avg_Confidence"] = summary_df["Avg_Confidence"].round(2)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: LIVE MYSQL TELEMETRY & ENVIRONMENTAL PARAMETER ANALYSIS
# ==============================================================================
with tab2:
    st.subheader("📡 Remote Field Telemetry (`SoilData` MySQL Database)")

    ctrl_col1, ctrl_col2 = st.columns([1, 3])
    with ctrl_col1:
        row_limit = st.selectbox("Records to Fetch", [50, 100, 250, 500, 1000], index=2)
    with ctrl_col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Live Database Records", type="primary"):
            st.rerun()

    if db_error:
        st.error(f"❌ **MySQL Connection Error:** {db_error}")
        st.info("Verify that your Hostinger Remote MySQL allows incoming connections from your current IP address (`%` wildcard in Remote MySQL settings).")
    elif env_df.empty:
        st.warning("Connected successfully, but the `SoilData` table currently has no records.")
    else:
        display_df = env_df.tail(row_limit).reset_index(drop=True)
        latest = display_df.iloc[-1]
        prev = display_df.iloc[-2] if len(display_df) > 1 else latest

        st.markdown(f"**Last Synced Timestamp:** `{latest['DateTime']}`")
        k1, k2, k3 = st.columns(3)

        temp_val = latest["EnvarmentTemp"]
        humi_val = latest["EnvarmentHumi"]
        soil_val = latest["SoilMoisture"]

        k1.metric(
            "Environment Temperature",
            f"{temp_val:.1f} °C" if pd.notna(temp_val) else "N/A",
            delta=f"{temp_val - prev['EnvarmentTemp']:+.1f} °C" if pd.notna(temp_val) and pd.notna(prev["EnvarmentTemp"]) else None
        )
        k2.metric(
            "Environment Humidity",
            f"{humi_val:.1f} %" if pd.notna(humi_val) else "N/A",
            delta=f"{humi_val - prev['EnvarmentHumi']:+.1f} %" if pd.notna(humi_val) and pd.notna(prev["EnvarmentHumi"]) else None
        )
        k3.metric(
            "Soil Moisture",
            f"{soil_val:.1f} %" if pd.notna(soil_val) else "N/A",
            delta=f"{soil_val - prev['SoilMoisture']:+.1f} %" if pd.notna(soil_val) and pd.notna(prev["SoilMoisture"]) else None
        )

        st.divider()

        # --- ANALYZE ENVIRONMENT PARAMETERS OPTION ---
        st.subheader("🧠 Analyze Environment Parameters")
        analysis_mode = st.radio(
            "Select Analysis View:",
            ["Real-Time Risk & Agronomic Assessment", "Historical Trend Charts", "Statistical Summary & Raw Data"],
            horizontal=True
        )

        if analysis_mode == "Real-Time Risk & Agronomic Assessment":
            st.markdown("#### Automated Pest & Crop Microclimate Diagnosis")

            if pd.notna(temp_val):
                if 24 <= temp_val <= 32:
                    st.warning(f"🌡️ **Temperature ({temp_val:.1f} °C):** Prime breeding zone for **Aphids, Whiteflies, and Armyworms** (24°C–32°C accelerates insect life cycles).")
                elif temp_val > 32:
                    st.error(f"🔥 **Temperature ({temp_val:.1f} °C):** High heat stress detected. Increases **Spider Mite** outbreak risk and crop dehydration.")
                else:
                    st.info(f"❄️ **Temperature ({temp_val:.1f} °C):** Cooler conditions; insect reproduction rate is naturally slowed.")

            if pd.notna(humi_val):
                if humi_val >= 75:
                    st.error(f"💧 **Humidity ({humi_val:.1f} %):** High humidity (>75%) creates severe risk for **Whiteflies, Leaf Miners, and fungal leaf blight**.")
                elif 50 <= humi_val < 75:
                    st.success(f"✅ **Humidity ({humi_val:.1f} %):** Balanced relative humidity for healthy crop transpiration.")
                else:
                    st.warning(f"🏜️ **Humidity ({humi_val:.1f} %):** Low humidity (<50%) favors dry-weather pests like **Thrips and Mites**.")

            if pd.notna(soil_val):
                if soil_val < 30:
                    st.error(f"🌱 **Soil Moisture ({soil_val:.1f} %):** Soil is dry. Water-stressed plants emit stress volatiles that attract sap-sucking pests — **Irrigation Recommended**.")
                elif soil_val > 80:
                    st.warning(f"🌊 **Soil Moisture ({soil_val:.1f} %):** Over-saturated soil increases root-rot and soil-borne larvae risk.")
                else:
                    st.success(f"✅ **Soil Moisture ({soil_val:.1f} %):** Optimal root-zone moisture level.")

            risk_score = 0
            if pd.notna(temp_val) and 24 <= temp_val <= 33:
                risk_score += 40
            if pd.notna(humi_val) and humi_val >= 70:
                risk_score += 40
            if pd.notna(soil_val) and (soil_val < 30 or soil_val > 80):
                risk_score += 20

            st.markdown("#### Overall Microclimate Pest Outbreak Risk Index")
            st.progress(risk_score / 100)
            st.write(f"**Calculated Environmental Risk Score:** `{risk_score} / 100` — " + (
                "**HIGH RISK** (Inspect leaves immediately in Tab 1)" if risk_score >= 70 else
                "**MODERATE RISK** (Routine monitoring advised)" if risk_score >= 40 else
                "**LOW RISK** (Environment unfavorable for rapid pest spread)"
            ))

        elif analysis_mode == "Historical Trend Charts":
            c1, c2 = st.columns(2)
            with c1:
                fig_temp_humi = px.line(
                    display_df,
                    x="DateTime",
                    y=["EnvarmentTemp", "EnvarmentHumi"],
                    title="Temperature (°C) & Humidity (%) Over Time",
                    labels={"value": "Reading", "variable": "Parameter"}
                )
                st.plotly_chart(fig_temp_humi, use_container_width=True)

            with c2:
                fig_soil = px.area(
                    display_df,
                    x="DateTime",
                    y="SoilMoisture",
                    title="Soil Moisture (%) Over Time",
                    color_discrete_sequence=["#2e7d32"]
                )
                st.plotly_chart(fig_soil, use_container_width=True)

        else:
            st.markdown("#### Statistical Summary (Min / Avg / Max)")
            stats_df = display_df[["SoilMoisture", "EnvarmentTemp", "EnvarmentHumi"]].describe().loc[["count", "mean", "min", "max"]].round(2)
            st.dataframe(stats_df, use_container_width=True)

            st.markdown("#### Raw Database Records (`SoilData`)")
            st.dataframe(display_df.sort_values("DateTime", ascending=False), use_container_width=True, hide_index=True)
