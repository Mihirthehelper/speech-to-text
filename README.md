# Streamlit Speech-to-Text Demo

A simple Streamlit app to transcribe uploaded audio using OpenAI's open-source Whisper model (small). The app converts audio to 16 kHz mono, runs transcription, displays editable timestamped segments, and lets you download the transcript as .txt or .srt.

Features
- Upload audio files (wav, mp3, m4a, webm)
- Convert audio to 16 kHz mono
- Transcribe using Whisper (model selectable)
- Show editable transcript with timestamps (segments)
- Export transcript as .txt or .srt

Quickstart (local)
1. Create and activate a virtual environment:
   - python -m venv venv
   - source venv/bin/activate  (macOS / Linux)
   - venv\Scripts\activate     (Windows)
2. Install dependencies:
   - pip install -r requirements.txt
   - Install ffmpeg (required by pydub). On macOS: `brew install ffmpeg`. On Ubuntu: `sudo apt install ffmpeg`
3. Run the app:
   - streamlit run streamlit_app.py

Notes
- Default model: `small` — a good balance between speed and quality for demos. For local CPU usage, `tiny` or `base` is faster. Larger models (medium/large) require more RAM and GPU to be practical.
- If you want a lighter offline-only approach, I can provide a Vosk-based variant.

Repository structure
- streamlit_app.py      (main app)
- requirements.txt
- src/
  - audio_utils.py
- assets/
  - sample_audio_download.py (script to download a sample file)
- .gitignore
- README.md

Deployment
- Streamlit Cloud: Connect repo and point the app to `streamlit_app.py` and set the required packages.
- If you use larger models, prefer running on a machine with GPU or switch to an API-based transcription.

Want me to:
- push this scaffold into a git repo for you? (I can prepare a PR if you give a repo), or
- add a Vosk/offline variant or an API-based variant?
