"""Fragrance sniff-test facial emotion tracker."""

import queue
import threading
import time

import av
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Fragrance Emotion Tracker", layout="wide")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
ANALYZE_EVERY_N_FRAMES = 15

if "log" not in st.session_state:
    st.session_state.log = []
if "session_start" not in st.session_state:
    st.session_state.session_start = None
if "respondent_id" not in st.session_state:
    st.session_state.respondent_id = ""


class EmotionProcessor:
    """Analyze frames off the UI thread and publish results safely."""

    def __init__(self):
        self.frame_count = 0
        self.last_result = None
        self.results = queue.Queue()
        self._lock = threading.Lock()
        self.last_error = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                result = DeepFace.analyze(
                    img,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv",
                    silent=True,
                )
                face = result[0] if isinstance(result, list) else result
                scores = face.get("emotion", {})
                dominant = face.get("dominant_emotion", "n/a")
                item = {
                    "captured_at": time.time(),
                    "dominant": dominant,
                    **{e: round(float(scores.get(e, 0.0)), 2) for e in EMOTIONS},
                }
                with self._lock:
                    self.last_result = item
                    self.last_error = None
                self.results.put(item)
            except Exception as exc:
                # Keep the video alive, but expose the error in the UI.
                with self._lock:
                    self.last_error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            latest = self.last_result
        if latest:
            import cv2

            cv2.putText(
                img,
                latest["dominant"],
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("🌸 Fragrance Sniff-Test — Facial Emotion Tracker")
st.caption(
    "Live webcam capture and local DeepFace analysis. Allow camera access when "
    "your browser asks, then start the exposure timer."
)

col_a, col_b = st.columns(2)
with col_a:
    st.session_state.respondent_id = st.text_input(
        "Respondent ID", value=st.session_state.respondent_id
    )
with col_b:
    st.text_input("Stimulus / fragrance being tested")

left, right = st.columns(2)
with left:
    st.subheader("Webcam feed")
    ctx = webrtc_streamer(
        key="emotion-tracker",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    b1, b2, b3 = st.columns(3)
    if b1.button("▶ Mark exposure start", use_container_width=True):
        st.session_state.session_start = time.time()
        st.session_state.log = []
        st.success("Exposure marked at t=0. Readings will appear on the right.")
    if b2.button("⏹ Stop logging", use_container_width=True):
        st.session_state.session_start = None
    if b3.button("🗑 Clear log", use_container_width=True):
        st.session_state.session_start = None
        st.session_state.log = []

    if not ctx.state.playing:
        st.info("Click START in the webcam box and allow camera permission.")


def drain_results():
    processor = ctx.video_processor
    if processor is None:
        return None
    while True:
        try:
            item = processor.results.get_nowait()
        except queue.Empty:
            break
        started = st.session_state.session_start
        if started is not None and item["captured_at"] >= started:
            row = {"t_sec": round(item["captured_at"] - started, 2)}
            row.update({k: v for k, v in item.items() if k != "captured_at"})
            st.session_state.log.append(row)
    with processor._lock:
        return processor.last_error


with right:
    st.subheader("Emotion timeline")

    @st.fragment(run_every=1.0)
    def render_timeline():
        error = drain_results()
        if error:
            st.error(f"Analysis error: {error}")
        if st.session_state.log:
            df = pd.DataFrame(st.session_state.log)
            st.line_chart(df.set_index("t_sec")[EMOTIONS])
            st.dataframe(df, use_container_width=True, height=250)
            csv = df.to_csv(index=False).encode("utf-8")
            name = st.session_state.respondent_id or "respondent"
            st.download_button(
                "Download CSV", csv, f"emotion_log_{name}.csv", "text/csv"
            )
        else:
            st.info("Start the webcam, then click Mark exposure start.")

    render_timeline()

st.warning(
    "Prototype only—not a validated psychometric instrument. Facial video is "
    "biometric data; obtain informed consent and complete the required privacy review."
)
