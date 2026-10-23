import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
import pandas as pd
import plotly.express as px

from PIL import Image
from pathlib import Path
from io import BytesIO
import sys
import time

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detector import ObjectDetector


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    background: #172033;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #28364d;
}

div.stButton > button,
div.stDownloadButton > button {
    width: 100%;
    border-radius: 8px;
    font-weight: 600;
    min-height: 42px;
}

section[data-testid="stSidebar"] {
    background-color: #10192b;
}

h1, h2, h3 {
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

@st.cache_resource
def load_detector():
    return ObjectDetector(model_name="yolo11n.pt")


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🎯 VisionTrack AI")
    st.caption("INTELLIGENT COMPUTER VISION")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Image Detection",
            "Video Tracking",
            "About Project"
        ]
    )

    st.markdown("---")
    st.subheader("⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Higher values show only more confident detections."
    )

    iou = st.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Controls overlap handling during object detection."
    )

    st.markdown("---")
    st.caption("YOLO • OpenCV • Streamlit")


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("🎯 VisionTrack AI")

st.write(
    "Intelligent Object Detection & Multi-Object Tracking Platform"
)

st.markdown("---")


# --------------------------------------------------
# IMAGE DETECTION
# --------------------------------------------------

if page == "Image Detection":

    st.subheader("🖼️ AI Image Detection")

    st.write(
        "Upload an image to detect and classify objects "
        "using a real YOLO deep-learning model."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown("### Uploaded Image")

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

            if st.button(
                "🚀 Run Object Detection",
                type="primary",
                key="run_detection"
            ):

                with st.spinner(
                    "Loading YOLO and analyzing the image..."
                ):

                    detector = load_detector()

                    start_time = time.perf_counter()

                    result = detector.detect(
                        image,
                        confidence=confidence,
                        iou=iou
                    )

                    elapsed_time = time.perf_counter() - start_time

                    detections = detector.get_detections(result)

                    annotated_image = detector.annotate(result)

                    st.session_state["detections"] = detections
                    st.session_state["annotated_image"] = annotated_image
                    st.session_state["processed_image_name"] = uploaded_file.name
                    st.session_state["processing_time"] = elapsed_time

        except Exception as error:
            st.error(f"Unable to process this image: {error}")

    # Display results only when detection has completed.
    if (
        "annotated_image" in st.session_state
        and "detections" in st.session_state
    ):

        detections = st.session_state["detections"]
        annotated_image = st.session_state["annotated_image"]

        st.markdown("---")
        st.subheader("🔍 Detection Results")

        original_column, result_column = st.columns(2)

        with original_column:
            st.markdown("#### Original Image")
            st.image(
                image,
                use_container_width=True
            )

        with result_column:
            st.markdown("#### Annotated Image")
            st.image(
                annotated_image,
                use_container_width=True
            )

        df = pd.DataFrame(detections)

        total_objects = len(detections)

        unique_classes = (
            df["class"].nunique()
            if not df.empty
            else 0
        )

        average_confidence = (
            df["confidence"].mean() * 100
            if not df.empty
            else 0
        )

        processing_time = st.session_state.get(
            "processing_time",
            0
        )

        st.markdown("---")
        st.subheader("📊 Detection Analytics")

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Objects Detected",
            total_objects
        )

        metric2.metric(
            "Object Categories",
            unique_classes
        )

        metric3.metric(
            "Average Confidence",
            f"{average_confidence:.1f}%"
        )

        metric4.metric(
            "Processing Time",
            f"{processing_time:.2f} sec"
        )

        if not df.empty:

            st.markdown("### 📋 Detection Details")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### 📈 Objects by Category")

            chart_data = (
                df.groupby("class")
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
            )

            figure = px.bar(
                chart_data,
                x="class",
                y="Count",
                text="Count",
                title="Detected Objects by Category",
                labels={
                    "class": "Object Category",
                    "Count": "Number of Objects"
                }
            )

            figure.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                plot_bgcolor="#0b1120"
            )

            st.plotly_chart(
                figure,
                use_container_width=True
            )

            st.markdown("### 📥 Download Results")

            download_column1, download_column2 = st.columns(2)

            csv_data = df.to_csv(index=False).encode("utf-8")

            with download_column1:
                st.download_button(
                    "📄 Download CSV Report",
                    data=csv_data,
                    file_name="visiontrack_detections.csv",
                    mime="text/csv"
                )

            image_buffer = BytesIO()

            Image.fromarray(annotated_image).save(
                image_buffer,
                format="PNG"
            )

            with download_column2:
                st.download_button(
                    "🖼️ Download Annotated Image",
                    data=image_buffer.getvalue(),
                    file_name="visiontrack_result.png",
                    mime="image/png"
                )

        else:
            st.warning(
                "No objects were detected. Try lowering "
                "the confidence threshold in the sidebar."
            )


# --------------------------------------------------
# VIDEO TRACKING PAGE
# --------------------------------------------------

elif page == "Video Tracking":

    st.subheader("🎥 Real-Time Object Detection & Tracking")
    st.write(
        "Use your webcam to detect objects in real time. YOLO draws boxes "
        "around supported objects, while ByteTrack attempts to keep their "
        "tracking IDs consistent between frames."
    )

    st.info(
        "Click START below and allow camera permission if your browser asks. "
        "For best results, run this app on localhost in a browser that has "
        "permission to use your camera."
    )

    class VisionTrackProcessor(VideoProcessorBase):
        def __init__(self, confidence_threshold=0.25, iou_threshold=0.45):
            self.detector = ObjectDetector(model_name=str(ROOT / "yolo11n.pt"))
            self.confidence = confidence_threshold
            self.iou = iou_threshold
            self.frame_count = 0
            self.last_detections = []

        def recv(self, frame):
            image = frame.to_ndarray(format="bgr24")
            result = self.detector.track(
                image,
                confidence=self.confidence,
                iou=self.iou
            )
            self.last_detections = self.detector.get_detections(result)
            self.frame_count += 1
            annotated_frame = result.plot()  # OpenCV BGR frame
            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="visiontrack-live-camera",
        video_processor_factory=lambda: VisionTrackProcessor(confidence, iou),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.caption(
        f"Current settings — confidence: {confidence:.2f} | IoU: {iou:.2f}"
    )
    if webrtc_ctx.state.playing:
        st.success("Webcam is active. Look at the video above for live detections.")
        st.caption(
            "Note: tracking IDs are provided when the tracker assigns them; "
            "they may change if an object disappears or tracking is interrupted."
        )
    else:
        st.warning("The camera is not running yet. Click START in the camera panel above.")


# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.subheader("About VisionTrack AI")

    st.write("""
    VisionTrack AI is a computer-vision project that uses a YOLO model
    to detect objects in uploaded images and track objects from a live
    webcam stream using ByteTrack through Streamlit WebRTC.

    The image page provides detection analytics and downloadable reports.
    """)

    st.markdown("### 🛠️ Technologies Used")

    st.markdown("""
    - **Python:** Main programming language
    - **YOLO:** Deep-learning object detection
    - **OpenCV:** Computer-vision and image-processing operations
    - **Streamlit:** Interactive dashboard
    - **Pandas:** Detection data and CSV reports
    - **Plotly:** Interactive charts
    - **Pillow:** Image handling and downloads
    - **Streamlit WebRTC:** Webcam video streaming
    - **ByteTrack:** Multi-object tracking support
    """)

    st.markdown("### 🚀 Possible Future Improvements")
    st.markdown("""
    - Display live object counts and tracking tables
    - Save annotated video clips
    - Show object movement trajectories
    - Add tracking analytics and performance metrics
    """)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "VisionTrack AI | Intelligent Computer Vision Platform"
)

