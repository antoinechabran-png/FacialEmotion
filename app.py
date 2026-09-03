"""Live fragrance-test facial emotion tracker built with Streamlit."""
from __future__ import annotations

import queue
import platform
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import Any

import av
import pandas as pd
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Fragrance Emotion Tracker", page_icon="🌸", layout="wide")

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
ANALYZE_EVERY_N_FRAMES = 15
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turn:openrelay.metered.ca:443?transport=tcp",
            ],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
}

if "emotion_log" not in st.session_state:
    st.session_state.emotion_log = []
if "exposure_started_at" not in st.session_state:
    st.session_state.exposure_started_at = None
if "respondent_id" not in st.session_state:
    st.session_state.respondent_id = ""
if "stimulus" not in st.session_state:
    st.session_state.stimulus = ""


class AppDiagnostics:
    """Thread-safe, bounded diagnostic log shared across Streamlit reruns."""

    def __init__(self) -> None:
        self._lines: deque[str] = deque(maxlen=500)
        self._lock = threading.Lock()

    def write(self, level: str, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        line = f"{timestamp} [{level}] {message}"
        with self._lock:
            self._lines.append(line)
        # Streamlit Community Cloud captures stdout in Manage app > Logs.
        print(line, file=sys.stdout, flush=True)

    def text(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()


@st.cache_resource
def get_diagnostics() -> AppDiagnostics:
    diagnostics = AppDiagnostics()
    diagnostics.write("INFO", "Application diagnostics initialized")
    diagnostics.write("INFO", f"Python={sys.version.replace(chr(10), ' ')}")
    diagnostics.write("INFO", f"Platform={platform.platform()}")
    diagnostics.write("INFO", f"Streamlit={st.__version__}")
    try:
        import cv2
        import deepface
        import tensorflow as tf
        import streamlit_webrtc

        diagnostics.write("INFO", f"OpenCV={cv2.__version__}")
        diagnostics.write("INFO", f"DeepFace={getattr(deepface, '__version__', 'unknown')}")
        diagnostics.write("INFO", f"TensorFlow={tf.__version__}")
        diagnostics.write(
            "INFO", f"streamlit-webrtc={getattr(streamlit_webrtc, '__version__', 'unknown')}"
        )
    except Exception:
        diagnostics.write("ERROR", "Version inspection failed:\n" + traceback.format_exc())
    return diagnostics


DIAGNOSTICS = get_diagnostics()


class EmotionProcessor:
    """Analyze frames off the UI thread without accessing session_state."""

    def __init__(self, diagnostics: AppDiagnostics) -> None:
        self.diagnostics = diagnostics
        self.frame_number = 0
        self.results: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100)
        self._state_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._last_emotion = ""
        self._last_error = ""
        self._started_at = time.perf_counter()
        self.diagnostics.write("INFO", "EmotionProcessor created; waiting for video frames")

    def _publish(self, result: dict[str, Any]) -> None:
        try:
            self.results.put_nowait(result)
        except queue.Full:
            try:
                self.results.get_nowait()
            except queue.Empty:
                pass
            self.results.put_nowait(result)

    def status(self) -> tuple[str, str]:
        with self._state_lock:
            return self._last_emotion, self._last_error

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_number += 1
        if self.frame_number == 1:
            height, width = image.shape[:2]
            self.diagnostics.write(
                "INFO", f"First video frame received: {width}x{height}, format=bgr24"
            )

        should_analyze = self.frame_number % ANALYZE_EVERY_N_FRAMES == 0
        # async_processing may overlap recv calls; serialize heavy inference.
        if should_analyze and self._inference_lock.acquire(blocking=False):
            inference_started = time.perf_counter()
            try:
                self.diagnostics.write(
                    "DEBUG", f"DeepFace inference started at frame {self.frame_number}"
                )
                analysis = DeepFace.analyze(
                    img_path=image,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True,
                    silent=True,
                )
                face = analysis[0] if isinstance(analysis, list) else analysis
                scores = face.get("emotion", {})
                dominant = str(face.get("dominant_emotion", "unknown"))
                result = {
                    "captured_at": time.time(),
                    "dominant": dominant,
                    **{
                        emotion: round(float(scores.get(emotion, 0.0)), 2)
                        for emotion in EMOTIONS
                    },
                }
                self._publish(result)
                with self._state_lock:
                    self._last_emotion = dominant
                    self._last_error = ""
                elapsed = time.perf_counter() - inference_started
                self.diagnostics.write(
                    "INFO",
                    f"DeepFace inference succeeded in {elapsed:.2f}s; dominant={dominant}",
                )
            except Exception as exc:
                details = traceback.format_exc()
                with self._state_lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                self.diagnostics.write("ERROR", "DeepFace inference failed:\n" + details)
            finally:
                self._inference_lock.release()

        last_emotion, _ = self.status()
        if last_emotion:
            import cv2

            cv2.putText(
                image, last_emotion, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 255, 0), 2, cv2.LINE_AA,
            )
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def collect_results(processor: EmotionProcessor | None) -> str:
    """Move queued results to Streamlit state on the UI thread."""
    if processor is None:
        return ""
    while True:
        try:
            item = processor.results.get_nowait()
        except queue.Empty:
            break
        started_at = st.session_state.exposure_started_at
        if started_at is None or item["captured_at"] < started_at:
            continue
        row = {
            "t_sec": round(item["captured_at"] - started_at, 2),
            "dominant": item["dominant"],
        }
        row.update({emotion: item[emotion] for emotion in EMOTIONS})
        st.session_state.emotion_log.append(row)
    _, error = processor.status()
    return error


def safe_filename(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    return cleaned.strip("_") or "respondent"


st.title("🌸 Fragrance Sniff-Test — Facial Emotion Tracker")
st.caption(
    "Click START in the camera panel, allow camera access, and then mark the "
    "exposure start. The first DeepFace result can take a little longer."
)

if "diagnostic_header_written" not in st.session_state:
    DIAGNOSTICS.write("INFO", "New browser session connected to the Streamlit app")
    st.session_state.diagnostic_header_written = True

identity_col, stimulus_col = st.columns(2)
with identity_col:
    st.session_state.respondent_id = st.text_input(
        "Respondent ID", value=st.session_state.respondent_id
    )
with stimulus_col:
    st.session_state.stimulus = st.text_input(
        "Stimulus / fragrance being tested", value=st.session_state.stimulus
    )

camera_col, results_col = st.columns(2)
with camera_col:
    st.subheader("Webcam feed")
    context = webrtc_streamer(
        key="emotion-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 24, "max": 30},
            },
            "audio": False,
        },
        video_processor_factory=lambda: EmotionProcessor(DIAGNOSTICS),
        async_processing=True,
        video_html_attrs={
            "autoPlay": True,
            "controls": True,
            "muted": True,
            "style": {"width": "100%"},
        },
    )

    start_col, stop_col, clear_col = st.columns(3)
    if start_col.button("▶ Mark exposure", use_container_width=True):
        if context.state.playing:
            st.session_state.exposure_started_at = time.time()
            st.session_state.emotion_log = []
            DIAGNOSTICS.write("INFO", "User marked exposure start; emotion log cleared")
            st.success("Exposure timer started at t=0.")
        else:
            DIAGNOSTICS.write("WARNING", "Exposure button clicked while camera disconnected")
            st.error("Start the webcam and allow camera access first.")
    if stop_col.button("⏹ Stop logging", use_container_width=True):
        st.session_state.exposure_started_at = None
        DIAGNOSTICS.write("INFO", "User stopped emotion logging")
        st.info("Logging stopped. Existing readings are preserved.")
    if clear_col.button("🗑 Clear", use_container_width=True):
        st.session_state.exposure_started_at = None
        st.session_state.emotion_log = []
        DIAGNOSTICS.write("INFO", "User cleared emotion log")

    if context.state.playing:
        st.success("Camera connected.")
    else:
        st.info(
            "Camera disconnected. Click START above and choose Allow when the "
            "browser requests camera permission."
        )

    with st.expander("Camera immediately disconnects?"):
        st.markdown(
            """
1. Click the lock/camera icon beside the browser address and select **Allow**.
2. Reload the page after changing camera permission.
3. Close Teams, Zoom, or other software that may be using the camera.
4. Temporarily disable a VPN or strict browser privacy extension.
5. Try Chrome or Edge on HTTPS, outside an embedded preview.
            """
        )

with results_col:
    st.subheader("Emotion timeline")

    @st.fragment(run_every=1.0)
    def render_results() -> None:
        state_now = "playing" if context.state.playing else "disconnected"
        previous_state = st.session_state.get("last_camera_state")
        if state_now != previous_state:
            DIAGNOSTICS.write(
                "INFO", f"WebRTC state changed: {previous_state or 'unknown'} -> {state_now}"
            )
            st.session_state.last_camera_state = state_now

        error = collect_results(context.video_processor)
        if error:
            st.error(f"DeepFace analysis error: {error}")

        log = st.session_state.emotion_log
        if not log:
            if st.session_state.exposure_started_at is None:
                st.info("Connect the camera, then click Mark exposure.")
            else:
                st.info("Waiting for the first emotion reading…")
            return

        data = pd.DataFrame(log)
        st.line_chart(data.set_index("t_sec")[list(EMOTIONS)])
        st.dataframe(data, use_container_width=True, height=260)
        csv_data = data.to_csv(index=False).encode("utf-8")
        respondent = safe_filename(st.session_state.respondent_id)
        st.download_button(
            "Download CSV", data=csv_data,
            file_name=f"emotion_log_{respondent}.csv", mime="text/csv",
            use_container_width=True,
        )

    render_results()

st.subheader("Diagnostic log")
st.caption(
    "No video images are recorded. This log contains connection state, software "
    "versions, processing milestones, and error tracebacks."
)

@st.fragment(run_every=1.0)
def render_diagnostics() -> None:
    diagnostic_text = DIAGNOSTICS.text()
    st.code(diagnostic_text or "No diagnostic events yet.", language="text")
    log_col, clear_log_col = st.columns([1, 1])
    log_col.download_button(
        "Download diagnostic log",
        data=diagnostic_text.encode("utf-8"),
        file_name="facial_emotion_diagnostics.log",
        mime="text/plain",
        use_container_width=True,
    )
    if clear_log_col.button("Clear diagnostic log", use_container_width=True):
        DIAGNOSTICS.clear()
        DIAGNOSTICS.write("INFO", "Diagnostic log cleared by user")


render_diagnostics()

st.divider()
st.warning(
    "Prototype only—not a validated psychometric instrument. Facial images are "
    "biometric data. Obtain informed consent and complete the applicable privacy "
    "and data-protection review before testing respondents."
)
