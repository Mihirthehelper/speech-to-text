# Minimal audio utilities expected by streamlit_app.py
# - load_whisper_model(model_name="base")
# - convert_uploaded_file_to_wav(uploaded, target_sr=16000) -> str (path to wav)
# - transcribe_with_whisper(model, wav_path, **kwargs) -> dict (whisper result)
#
# Requires:
# - ffmpeg binary on PATH
# - openai-whisper package installed (whisper.load_model)
# - subprocess, tempfile, os

import subprocess
import tempfile
import os
from typing import Union, Tuple, Dict

# Lazy-import whisper to avoid import-time cost if not needed
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


def convert_uploaded_file_to_wav(
    uploaded: Union[bytes, str], target_sr: int = 16000
) -> str:
    """
    Convert uploaded audio (bytes or file path) to a temporary WAV file (mono, target_sr).
    Returns the path to the WAV file on disk. Caller should remove the file when done.
    """
    # Accept either raw bytes or a path (string)
    if isinstance(uploaded, bytes):
        in_path = _write_bytes_to_tempfile(uploaded)
    elif isinstance(uploaded, str):
        in_path = uploaded
    else:
        raise TypeError("uploaded must be bytes or a file path string")

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
    # Run ffmpeg; hide output unless it fails
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        # If we created a temp input file from bytes, remove it
        if isinstance(uploaded, bytes):
            try:
                os.remove(in_path)
            except Exception:
                pass
        # remove out file if present
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise RuntimeError("ffmpeg failed to convert uploaded audio to WAV") from e

    # remove temp input if it was created from bytes
    if isinstance(uploaded, bytes):
        try:
            os.remove(in_path)
        except Exception:
            pass

    return out_path


def transcribe_with_whisper(model, wav_path: str, **kwargs) -> Dict:
    """
    Transcribe the WAV file at wav_path using the given whisper model.
    Returns the full whisper result dict (contains 'text', segments, etc).
    Example:
      result = transcribe_with_whisper(model, wav_path, language="en", temperature=0.0)
      text = result["text"]
    """
    # whisper models accept file path directly
    try:
        result = model.transcribe(wav_path, **kwargs)
    except Exception as e:
        raise RuntimeError("Whisper transcription failed") from e
    return result
