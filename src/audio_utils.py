# Minimal audio utilities without pydub.
# Uses ffmpeg + soundfile to convert arbitrary audio to 16k mono WAV and read it.
# Requires: soundfile (pysoundfile), numpy, and ffmpeg binary available on PATH.

import subprocess
import tempfile
import os
from typing import Tuple, Union
import numpy as np
import soundfile as sf

def _write_bytes_to_tempfile(data: bytes) -> str:
    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        f.write(data)
        f.flush()
        return f.name
    finally:
        f.close()

def _ffmpeg_convert_to_wav(in_path: str, out_path: str, sample_rate: int = 16000) -> None:
    """
    Convert in_path (any ffmpeg-supported audio file) to WAV at sample_rate, mono.
    Output written to out_path.
    Raises CalledProcessError on failure.
    """
    cmd = [
        "ffmpeg",
        "-y",              # overwrite
        "-i", in_path,     # input file
        "-ar", str(sample_rate),  # audio sample rate
        "-ac", "1",        # mono
        "-f", "wav",
        out_path
    ]
    # We hide ffmpeg stdout/stderr to keep logs tidy; remove stdout/stderr args to debug.
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def load_audio_from_bytes(data: bytes, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Convert bytes of an audio file to a numpy float32 array and sample rate.
    Returns (audio, sample_rate). Audio is 1-D numpy float32 with values in [-1.0, 1.0].
    """
    in_path = _write_bytes_to_tempfile(data)
    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    try:
        _ffmpeg_convert_to_wav(in_path, out_path, sample_rate=target_sr)
        audio, sr = sf.read(out_path, dtype="float32")
        # sf.read may return (n_frames, 1) for mono; flatten to 1D
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio, sr
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except Exception:
                pass

def load_audio(source: Union[str, bytes], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio from a file path or raw bytes.
    - If `source` is bytes, it will be treated as file contents and piped through ffmpeg.
    - If `source` is a string path, it will be read by ffmpeg similarly.
    Returns (audio, sample_rate).
    """
    if isinstance(source, bytes):
        return load_audio_from_bytes(source, target_sr=target_sr)
    elif isinstance(source, str):
        # treat as file path
        out_fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(out_fd)
        try:
            _ffmpeg_convert_to_wav(source, out_path, sample_rate=target_sr)
            audio, sr = sf.read(out_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass
    else:
        raise TypeError("source must be bytes or str(file path)")
