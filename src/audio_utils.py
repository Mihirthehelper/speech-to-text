import whisper
from pydub import AudioSegment
import tempfile
import os
import json
from typing import List, Dict


_model_cache = {}


def load_whisper_model(model_name: str = "small"):
    """
    Load and cache a whisper model.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    model = whisper.load_model(model_name)
    _model_cache[model_name] = model
    return model


def convert_uploaded_file_to_wav(uploaded_file) -> str:
    """
    Take a Streamlit uploaded_file (a BytesIO), write to temp file, convert with pydub to 16k mono WAV,
    and return the temp wav filepath.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_tmp:
        in_tmp.write(uploaded_file.read())
        in_path = in_tmp.name

    # pydub auto-detects format from suffix
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_path = out_tmp.name
    audio.export(out_path, format="wav")
    # remove input temp
    try:
        os.remove(in_path)
    except Exception:
        pass
    return out_path


def transcribe_with_whisper(model, audio_file_path: str, language: str = None) -> Dict:
    """
    Transcribe with whisper and return the raw result dict (including segments).
    """
    options = {}
    if language:
        options["language"] = language
        options["task"] = "transcribe"
    # Use fp16=False on CPU to avoid errors
    result = model.transcribe(audio_file_path, **options, fp16=False)
    return result


def segments_to_text(segments: List[Dict]) -> str:
    return "\n".join([seg.get("text", "").strip() for seg in segments])


def seconds_to_srt_timestamp(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"


def segments_to_srt(segments: List[Dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = seconds_to_srt_timestamp(seg["start"])
        end = seconds_to_srt_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line
    return "\n".join(lines)
