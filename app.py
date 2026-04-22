import streamlit as st
import cv2
import numpy as np
import json
from detector.model import ObjectDetectionModel
from detector.inference import detect_objects
from detector.utils import load_labels
from detector.logger import setup_logger

# Logger
logger = setup_logger()

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DetectOBJ",
    layout="wide",
    page_icon="🎯"
)

# ================= CUSTOM CSS (🔥 HIGH-END UI) =================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    text-align: center;
    font-size: 3rem;
    background: linear-gradient(90deg, #00DBDE, #FC00FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.metric {
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1>🎯 DetectOBJ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Next-Gen AI Object Detection Dashboard</p>", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

labels = load_labels("models/labels.txt")

selected_classes = st.sidebar.multiselect("🎯 Filter Classes", labels, default=[])

mode = st.sidebar.radio("🎥 Select Mode", ["Image", "Webcam", "Video"])

# ================= MODEL =================
@st.cache_resource
def load_model():
    loader = ObjectDetectionModel(
        "models/frozen_inference_graph.pb",
        "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    )
    return loader.get_model()

model = load_model()

# ================= IMAGE MODE =================
if mode == "Image":

    st.markdown("### 🖼️ Image Detection")

    file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if file:
        with st.spinner("🔍 Detecting objects..."):
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)

            output, results, count, fps = detect_objects(
                model, image, labels, confidence, selected_classes
            )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Original")
            st.image(image, use_column_width=True)

        with col2:
            st.markdown("#### Detected")
            st.image(output, use_column_width=True)

        # Metrics
        col3, col4 = st.columns(2)

        with col3:
            st.markdown(f"<div class='card'><p class='metric'>⚡ FPS: {round(fps,2)}</p></div>", unsafe_allow_html=True)

        with col4:
            st.markdown(f"<div class='card'><p class='metric'>📊 Objects: {sum(count.values())}</p></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Detection Breakdown")
        st.json(count)

        # Save outputs
        cv2.imwrite("outputs/images/result.jpg", output)
        with open("outputs/results.json", "w") as f:
            json.dump(results, f)

        # Download buttons
        with open("outputs/images/result.jpg", "rb") as file_img:
            st.download_button("⬇️ Download Image", file_img, file_name="result.jpg")

        with open("outputs/results.json", "rb") as file_json:
            st.download_button("⬇️ Download JSON", file_json, file_name="results.json")

# ================= WEBCAM MODE =================
elif mode == "Webcam":

    st.markdown("### 📷 Live Webcam Detection")

    run = st.toggle("Start Camera")

    cap = cv2.VideoCapture(0)

    frame_window = st.image([])

    status = st.empty()

    while run:
        ret, frame = cap.read()

        if not ret:
            st.error("Camera not working")
            break

        output, _, count, fps = detect_objects(
            model, frame, labels, confidence, selected_classes
        )

        frame_window.image(output)

        status.markdown(f"""
        <div class='card'>
        ⚡ FPS: {round(fps,2)} <br>
        📊 Objects: {sum(count.values())}
        </div>
        """, unsafe_allow_html=True)

    cap.release()

# ================= VIDEO MODE =================
elif mode == "Video":

    st.markdown("### 🎥 Video Detection")

    file = st.file_uploader("Upload Video", type=["mp4"])

    if file:
        with st.spinner("Processing video..."):
            tfile = open("temp.mp4", "wb")
            tfile.write(file.read())

            cap = cv2.VideoCapture("temp.mp4")

            frame_window = st.image([])

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                output, _, _, _ = detect_objects(
                    model, frame, labels, confidence, selected_classes
                )

                frame_window.image(output)

            cap.release()

        st.success("✅ Video processing completed")