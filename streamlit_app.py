import streamlit as st
from src.audio_utils import (
    load_whisper_model,
    convert_uploaded_file_to_wav,
    transcribe_with_whisper,
    segments_to_srt,
    segments_to_text,
)
import tempfile
import os

st.set_page_config(page_title="Speech-to-Text Demo", layout="wide")

st.title("Speech-to-Text Demo (Whisper)")
st.markdown(
    "Upload an audio file (wav, mp3, m4a) and transcribe it using a local Whisper model. "
    "For deployment or heavy usage consider using a hosted API or smaller models."
)

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Whisper model", options=["tiny", "base", "small"], index=2)
    language = st.text_input("Language (leave blank for auto-detect)", value="")
    show_timestamps = st.checkbox("Show timestamps", value=True)
    run_button = st.button("Load model & transcribe")

st.info("Tip: if this is your first run the model will download and may take a minute.")

# File uploader
uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "webm", "ogg"])

# Button fallback for users who prefer explicit action
if run_button and not uploaded_file:
    st.warning("Please upload an audio file first.")
    st.stop()

model = None
if uploaded_file:
    # Convert and save a temporary wav file
    st.audio(uploaded_file, format="audio/*")
    tmp_wav = convert_uploaded_file_to_wav(uploaded_file)
    st.write("Prepared audio for transcription:", tmp_wav)

    # Load model (cached)
    with st.spinner(f"Loading Whisper ({model_name})..."):
        model = load_whisper_model(model_name)

    if st.button("Transcribe now") or run_button:
        with st.spinner("Transcribing..."):
            result = transcribe_with_whisper(model, tmp_wav, language=language or None)

        # Show whole transcript editable
        st.subheader("Transcript")
        full_text = segments_to_text(result.get("segments", []))
        edited_text = st.text_area("Edit transcript", value=full_text, height=300)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download TXT", edited_text, file_name="transcript.txt", mime="text/plain")
        with col2:
            srt_text = segments_to_srt(result.get("segments", []))
            st.download_button("Download SRT", srt_text, file_name="transcript.srt", mime="text/plain")

        if show_timestamps:
            st.subheader("Segment timestamps")
            for seg in result.get("segments", []):
                st.markdown(f"- **{seg['start']:.2f}s - {seg['end']:.2f}s** — {seg['text']}")

    # Clean up temp file on exit
    try:
        if tmp_wav and os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    except Exception:
        pass
else:
    st.info("Upload a file to begin. If you need a sample audio, run `python assets/sample_audio_download.py` to download one.")
