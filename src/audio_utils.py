# Minimal audio utilities expected by streamlit_app.py
# - load_whisper_model(model_name="base")
# - convert_uploaded_file_to_wav(uploaded, target_sr=16000) -> str (path to wav)
# - transcribe_with_whisper(model, wav_path, **kwargs) -> dict (whisper result)
# - segments_to_srt(segments) -> str (SRT formatted captions)
#
# Requirements:
# - ffmpeg binary on PATH
# - openai-whisper package installed (whisper.load_model)
# - subprocess, tempfile, os

import subprocess
import tempfile
import os
from typing import Union, Dict, List, Any


def load_whisper_model(model_name: str = "base"):
    """
    Load and return a Whisper model. Example: model = load_whisper_model("small")
    """
    try:
        import whisper
    except Exception as e:
        raise ImportError(
            "Failed to import whisper. Make sure openai-whisper is installed and compatible with the runtime."
        ) from e
    return whisper.load_model(model_name)


def _write_bytes_to_tempfile(data: bytes, suffix: str = "") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(data)
    return path


def _is_file_like(obj: object) -> bool:
    return hasattr(obj, "read") and callable(getattr(obj, "read"))


def convert_uploaded_file_to_wav(
    uploaded: Union[bytes, str, object], target_sr: int = 16000
) -> str:
    """
    Convert uploaded audio (bytes, file path or file-like object) to a temporary WAV file
    (mono, target_sr). Returns the path to the WAV file on disk. Caller should remove the file when done.
    Accepts:
      - bytes: raw file contents
      - str: path to a file already on disk
      - file-like object with .read() (e.g., Streamlit UploadedFile)
    """
    created_input = False

    # Decide input path
    if isinstance(uploaded, bytes):
        in_path = _write_bytes_to_tempfile(uploaded)
        created_input = True
    elif isinstance(uploaded, str):
        in_path = uploaded
    elif _is_file_like(uploaded):
        content = uploaded.read()
        # Some file-likes allow multiple reads; ensure bytes
        if isinstance(content, str):
            content = content.encode()
        in_path = _write_bytes_to_tempfile(content)
        created_input = True
    else:
        raise TypeError("uploaded must be bytes, a file path string, or a file-like object")

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        in_path,
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        "-f",
        "wav",
        out_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        # cleanup
        try:
            if created_input:
                os.remove(in_path)
        except Exception:
            pass
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise RuntimeError("ffmpeg failed to convert uploaded audio to WAV") from e

    # If we created a temp input file from bytes/file-like, remove it now
    if created_input:
        try:
            os.remove(in_path)
        except Exception:
            pass

    return out_path


def transcribe_with_whisper(model, wav_path: str, **kwargs) -> Dict[str, Any]:
    """
    Transcribe the WAV file at wav_path using the given whisper model.
    Returns the full whisper result dict (contains 'text', 'segments', etc).
    Example:
      result = transcribe_with_whisper(model, wav_path, language="en", temperature=0.0)
      text = result["text"]
    """
    try:
        result = model.transcribe(wav_path, **kwargs)
    except Exception as e:
        raise RuntimeError("Whisper transcription failed") from e
    return result


def _format_srt_timestamp(seconds: float) -> str:
    """
    Convert seconds (float) to SRT timestamp: 'HH:MM:SS,mmm'
    """
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int(round((seconds - int(seconds)) * 1000))
    # Fix rounding that can push msecs to 1000
    if msecs >= 1000:
        msecs -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            mins += 1
            if mins >= 60:
                mins -= 60
                hrs += 1
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Convert a list of Whisper segments to SRT formatted captions.
    Each segment must be a dict containing at least: 'start', 'end', 'text'.
    Returns the full SRT as a string.
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = seg.get("start", 0.0)
        end = seg.get("end", start + 0.0)
        text = seg.get("text", "").strip()

        start_ts = _format_srt_timestamp(float(start))
        end_ts = _format_srt_timestamp(float(end))
        lines.append(str(i))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")  # blank line between entries

    return "\n".join(lines)
