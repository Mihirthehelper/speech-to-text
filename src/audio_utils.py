# Minimal audio utilities expected by streamlit_app.py
# - load_whisper_model(model_name="base")
# - convert_uploaded_file_to_wav(uploaded, target_sr=16000) -> str (path to wav)
# - transcribe_with_whisper(model, wav_path, **kwargs) -> dict (whisper result)
# - segments_to_srt(segments) -> str (SRT formatted captions)
# - segments_to_text(segments, sep=" ") -> str (concatenated segment text)
#
# Strategy:
# - If the uploaded file is WAV (filename or header), read it with soundfile and resample in-Python (no ffmpeg).
# - Otherwise, try to convert with ffmpeg. If ffmpeg is not found, raise a helpful error instructing the user to upload WAV or install ffmpeg.

import subprocess
import tempfile
import os
from typing import Union, Dict, List, Any
import numpy as np
import soundfile as sf


def load_whisper_model(model_name: str = "base"):
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


def _is_wav_bytes(data: bytes) -> bool:
    # Quick check for RIFF header (common for WAV)
    return len(data) >= 12 and (data[:4] == b"RIFF" or data[:4] == b"RIFX")


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample 1-D numpy array from orig_sr to target_sr using linear interpolation.
    This is a simple, dependency-free resampler (not high-quality but sufficient for many cases).
    """
    if orig_sr == target_sr:
        return audio
    orig_len = audio.shape[0]
    new_len = int(np.round(orig_len * (target_sr / orig_sr)))
    if new_len <= 0:
        return np.array([], dtype=np.float32)
    orig_x = np.linspace(0, 1, orig_len)
    new_x = np.linspace(0, 1, new_len)
    resampled = np.interp(new_x, orig_x, audio).astype(np.float32)
    return resampled


def convert_uploaded_file_to_wav(
    uploaded: Union[bytes, str, object], target_sr: int = 16000
) -> str:
    """
    Convert uploaded audio (bytes, file path or file-like object) to a temporary WAV file
    (mono, target_sr). Returns the path to the WAV file on disk. Caller should remove the file when done.

    Behavior:
    - If the uploaded content is (or looks like) WAV, we read with soundfile and resample in-Python (no ffmpeg).
    - Otherwise we try ffmpeg. If ffmpeg is missing, we raise a helpful error directing the user to upload WAV or install ffmpeg.
    """
    created_input = False

    # Obtain raw bytes or an input path
    in_path = None
    data_bytes = None
    filename = None

    if isinstance(uploaded, bytes):
        data_bytes = uploaded
        created_input = True
    elif isinstance(uploaded, str):
        # already on disk
        in_path = uploaded
        filename = os.path.basename(in_path)
    elif _is_file_like(uploaded):
        # Streamlit's UploadedFile has .read() and .name
        try:
            data_bytes = uploaded.read()
        except Exception:
            # fallback: try to use the file object's name if present
            if getattr(uploaded, "name", None):
                in_path = uploaded.name
            else:
                raise
        filename = getattr(uploaded, "name", None)
        if data_bytes is not None:
            created_input = True
    else:
        raise TypeError("uploaded must be bytes, a file path string, or a file-like object")

    # If we have bytes and filename is not provided, try to detect WAV header
    looks_like_wav = False
    if data_bytes is not None:
        if filename and filename.lower().endswith(".wav"):
            looks_like_wav = True
        elif _is_wav_bytes(data_bytes):
            looks_like_wav = True

    # If data_bytes present and we need to make an input file path for soundfile, write it
    if data_bytes is not None and not looks_like_wav:
        # we still write bytes to a temp file to allow ffmpeg path if available
        in_path = _write_bytes_to_tempfile(data_bytes)
    elif data_bytes is not None and looks_like_wav:
        in_path = _write_bytes_to_tempfile(data_bytes)

    # If it's WAV (either detected or file extension), try to read with soundfile and resample
    if looks_like_wav or (in_path and in_path.lower().endswith(".wav")):
        try:
            audio, sr = sf.read(in_path, dtype="float32")
            # Convert multi-channel to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Resample if needed
            if sr != target_sr:
                audio = _resample_audio(audio, sr, target_sr)
                sr = target_sr
            # Write out a clean WAV file
            out_fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(out_fd)
            sf.write(out_path, audio, samplerate=sr, subtype="PCM_16")
            # cleanup input temp if created
            if created_input and in_path:
                try:
                    os.remove(in_path)
                except Exception:
                    pass
            return out_path
        except Exception as e:
            # If soundfile read fails, fall back to ffmpeg (if available) below
            pass

    # If not WAV or soundfile failed, try ffmpeg conversion
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ar", str(target_sr), "-ac", "1", "-f", "wav"]
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    cmd.append(out_path)

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        # ffmpeg binary not present
        # cleanup input if we created it
        if created_input and in_path:
            try:
                os.remove(in_path)
            except Exception:
                pass
        # Provide a clear, actionable error message
        raise RuntimeError(
            "ffmpeg is not available in the runtime. Please upload a WAV file (16 kHz mono is best), "
            "or install ffmpeg in your environment so the app can convert MP3/M4A/OGG files. "
            "On many systems you can install it with 'apt-get install ffmpeg' or similar."
        )
    except subprocess.CalledProcessError as e:
        # ffmpeg failed to convert
        if created_input and in_path:
            try:
                os.remove(in_path)
            except Exception:
                pass
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise RuntimeError("ffmpeg failed to convert uploaded audio to WAV") from e

    # remove input temp if created
    if created_input and in_path:
        try:
            os.remove(in_path)
        except Exception:
            pass

    return out_path


def transcribe_with_whisper(model, wav_path: str, **kwargs) -> Dict[str, Any]:
    try:
        result = model.transcribe(wav_path, **kwargs)
    except Exception as e:
        raise RuntimeError("Whisper transcription failed") from e
    return result


def _format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int(round((seconds - int(seconds)) * 1000))
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
        lines.append("")
    return "\n".join(lines)


def segments_to_text(segments: List[Dict[str, Any]], sep: str = " ") -> str:
    texts = []
    for seg in segments:
        t = seg.get("text", "")
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if t:
            texts.append(t)
    return sep.join(texts).strip()
