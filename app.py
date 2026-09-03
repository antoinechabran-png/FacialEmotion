"""
Fragrance Sniff-Test Emotion Tracker — Prototype
--------------------------------------------------
Captures webcam video in the browser, runs facial emotion analysis
(DeepFace) on periodic frames, and logs a timestamped emotion timeline
for a consumer test respondent. Designed as a starting point — not a
validated research instrument.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud: push this folder (app.py +
requirements.txt) to a GitHub repo and point Streamlit Cloud at it.
"""

import time
from collections import deque

import av
import numpy as np
import pandas as pd
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Fragrance Emotion Tracker", layout="wide")

# ---------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------
if "log" not in st.session_state:
    st.session_state.log = []  # list of dicts: {t, emotion, scores...}
if "session_start" not in st.session_state:
    st.session_state.session_start = None
if "respondent_id" not in st.session_state:
    st.session_state.respondent_id = ""

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
ANALYZE_EVERY_N_FRAMES = 10  # ~ every 1/3s at 30fps; tune for your CPU


# ---------------------------------------------------------------------
# Video processor: runs in a background thread per WebRTC frame
# ---------------------------------------------------------------------
class EmotionProcessor:
    def __init__(self):
        self.frame_count = 0
        self.last_result = None

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
                self.last_result = {"dominant": dominant, "scores": scores}

                if st.session_state.session_start is not None:
                    elapsed = time.time() - st.session_state.session_start
                    row = {"t_sec": round(elapsed, 2), "dominant": dominant}
                    row.update({e: round(scores.get(e, 0.0), 2) for e in EMOTIONS})
                    st.session_state.log.append(row)
            except Exception:
                pass  # no face detected in this frame — skip silently

        # Overlay the latest dominant emotion on the video feed
        if self.last_result:
            import cv2

            label = self.last_result["dominant"]
            cv2.putText(
                img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 255, 0), 2, cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.title("🌸 Fragrance Sniff-Test — Facial Emotion Tracker (Prototype)")
st.caption(
    "Live webcam capture + DeepFace emotion analysis. For piloting only — "
    "not a validated psychometric instrument. Get informed consent before "
    "recording anyone's face."
)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.session_state.respondent_id = st.text_input(
        "Respondent ID", value=st.session_state.respondent_id
    )
with col_b:
    stimulus_label = st.text_input("Stimulus / fragrance being tested", value="")

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("Webcam feed")
    ctx = webrtc_streamer(
        key="emotion-tracker",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    b1, b2, b3 = st.columns(3)
    if b1.button("▶ Mark exposure start", use_container_width=True):
        st.session_state.session_start = time.time()
        st.session_state.log = []
        st.success("Timer reset — exposure marked at t=0.")
    if b2.button("⏹ Stop / snapshot log", use_container_width=True):
        st.session_state.session_start = None
    if b3.button("🗑 Clear log", use_container_width=True):
        st.session_state.log = []

with right:
    st.subheader("Emotion timeline")
    if st.session_state.log:
        df = pd.DataFrame(st.session_state.log)
        st.line_chart(df.set_index("t_sec")[EMOTIONS])
        st.dataframe(df, use_container_width=True, height=250)

        csv = df.to_csv(index=False).encode("utf-8")
        fname = f"emotion_log_{st.session_state.respondent_id or 'respondent'}.csv"
        st.download_button("Download CSV", csv, fname, "text/csv")
    else:
        st.info("Click **Mark exposure start**, then let a face be visible "
                 "to the webcam to start logging emotion readings.")

st.markdown("---")
with st.expander("Notes on this prototype"):
    st.markdown(
        """
- Analyzes roughly every 10th frame to keep CPU load manageable — tune
  `ANALYZE_EVERY_N_FRAMES` for your hardware.
- Uses DeepFace's 7-class emotion model (angry, disgust, fear, happy,
  sad, surprise, neutral). For subtler reactions (e.g. a brief sniff
  response) an Action-Unit-based model like **py-feat** may capture
  more nuance than these coarse categories.
- Readings are per-frame and noisy — for real analysis, smooth over a
  rolling window and look at peaks/deltas relative to the exposure
  marker rather than single readings.
- Video frames are processed locally in this session and are not
  stored — only the emotion scores are logged. Confirm this matches
  your actual privacy requirements before using with real
  respondents; facial video is biometric data under GDPR, so informed
  consent and a data protection review are needed before rollout.
- Streamlit Community Cloud's free tier can be tight on CPU/RAM for
  TensorFlow-based DeepFace — for a 3,000-person panel this would
  need proper infra (e.g. your CMI sandbox deployment) rather than
  the free tier.
        """
    )
