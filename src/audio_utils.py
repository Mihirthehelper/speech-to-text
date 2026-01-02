"""
Audio utilities used by the Streamlit app.

Provides:
- load_whisper_model(model_name="base")
- convert_uploaded_file_to_wav(uploaded, target_sr=16000) -> str (path to wav)
- transcribe_with_whisper(model, wav_path, **kwargs) -> dict (whisper result)
- segments_to_srt(segments) -> str (SRT formatted captions)
- segments_to_text(segments, sep=" ") -> str (concatenated segment text)

Behavior:
- If uploaded content is WAV, reads with soundfile and resamples in-Python (no ffmpeg).
- Otherwise tries to convert with system ffmpeg, falling back to the pip package imageio-ffmpeg
  if available. If neither is available, raises a helpful RuntimeError.
"""

import os
import subprocess
import tempfile
import shutil
from typing import Union, Dict, List, Any

import numpy as np
import soundfile as sf


def load_whisper_model(model_name: str = "base"):
    """
    Lazy-load and return a Whisper model. Example: model = load_whisper_model("small")
    """
    try:
        import whisper  # type: ignore
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
    # Quick check for RIFF header used by WAV files
    return len(data) >= 12 and (data[:4] == b"RIFF" or data[:4] == b"RIFX")


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample 1-D numpy array from orig_sr to target_sr using linear interpolation.
    This is a simple, dependency-free resampler (not ultra high-quality but sufficient).
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    orig_len = audio.shape[0]
    new_len = int(np.round(orig_len * (target_sr / orig_sr)))
    if new_len <= 0 or orig_len == 0:
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

    Accepts:
      - bytes: raw file contents
      - str: path to a file already on disk
      - file-like object with .read() (e.g., Streamlit UploadedFile)
    """
    created_input = False

    # Determine input source
    in_path = None
    data_bytes = None
    filename = None

    if isinstance(uploaded, bytes):
        data_bytes = uploaded
        created_input = True
    elif isinstance(uploaded, str):
        in_path = uploaded
        filename = os.path.basename(in_path)
    elif _is_file_like(uploaded):
        # Streamlit's UploadedFile has .read() and often .name
        try:
            data_bytes = uploaded.read()
        except Exception:
            # Fallback to using an existing file path if provided on the object
            if getattr(uploaded, "name", None):
                in_path = uploaded.name
            else:
                raise
        filename = getattr(uploaded, "name", None)
        if data_bytes is not None:
            created_input = True
    else:
        raise TypeError("uploaded must be bytes, a file path string, or a file-like object")

    # Detect WAV-like content
    looks_like_wav = False
    if data_bytes is not None:
        if filename and filename.lower().endswith(".wav"):
            looks_like_wav = True
        elif _is_wav_bytes(data_bytes):
            looks_like_wav = True

    # If we have bytes, write to a temp file for either soundfile reading or ffmpeg
    if data_bytes is not None:
        in_path = _write_bytes_to_tempfile(data_bytes)

    # If WAV-like or path ends with .wav, try to read and resample with soundfile (no ffmpeg)
    try_soundfile = looks_like_wav or (in_path and in_path.lower().endswith(".wav"))
    if try_soundfile and in_path:
        try:
            audio, sr = sf.read(in_path, dtype="float32")
            # Convert multi-channel to mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Resample if necessary
            if sr != target_sr:
                audio = _resample_audio(audio, sr, target_sr)
                sr = target_sr
            # Write out a clean WAV file (PCM16)
            out_fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(out_fd)
            sf.write(out_path, audio, samplerate=sr, subtype="PCM_16")
            # cleanup input temp if we created it
            if created_input and in_path:
                try:
                    os.remove(in_path)
                except Exception:
                    pass
            return out_path
        except Exception:
            # If soundfile fails, fall through to ffmpeg attempt below
            pass

    # If not WAV or soundfile read failed, use ffmpeg.
    # Prefer system ffmpeg; if missing, fall back to imageio-ffmpeg bundled binary.
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg as _iioff  # type: ignore

            # imageio-ffmpeg exposes a function to get the binary path.
            # Older versions: _iioff.get_ffmpeg_exe(); newer aliases may vary.
            try:
                ffmpeg_exe = _iioff.get_ffmpeg_exe()
            except AttributeError:
                # try alternate name
                ffmpeg_exe = _iioff.get_exe()
        except Exception:
            ffmpeg_exe = None

    if ffmpeg_exe is None:
        # cleanup input if we created it
        if created_input and in_path:
            try:
                os.remove(in_path)
            except Exception:
                pass
        raise RuntimeError(
            "ffmpeg is not available in the runtime. Please upload a WAV file (16 kHz mono is best), "
            "or add the 'imageio-ffmpeg' package to requirements so a bundled ffmpeg will be available."
        )

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)

    cmd = [
        ffmpeg_exe,
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


def transcribe_with_whisper(model, wav_path: str, target_sr: int = 16000, **kwargs) -> Dict[str, Any]:
    """
    Transcribe the audio using a Whisper model without invoking Whisper's ffmpeg-based loader.
    - If wav_path is a file path (string) we read it with soundfile and resample (if necessary),
      then pass the numpy array to model.transcribe to avoid ffmpeg.
    - If wav_path is already an array/tensor, we pass it through to model.transcribe.
    """
    try:
        # If user passed a path, load it with soundfile so Whisper won't call ffmpeg.
        if isinstance(wav_path, str):
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != target_sr:
                audio = _resample_audio(audio, sr, target_sr)
            result = model.transcribe(audio, **kwargs)
        else:
            # assume wav_path is already a numpy array or torch tensor
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
    # Fix rounding overflow
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
    Each segment is a dict with keys 'start', 'end', 'text' (typical whisper segments).
    """
    lines: List[str] = []
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
    """
    Convert Whisper segments to plain concatenated text.
    """
    texts: List[str] = []
    for seg in segments:
        t = seg.get("text", "")
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if t:
            texts.append(t)
    return sep.join(texts).strip()
