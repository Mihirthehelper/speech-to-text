"""
Small helper to download a sample audio file for testing.
"""
import requests
import os

URL = "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars60.wav"
OUT = "assets/sample.wav"

os.makedirs("assets", exist_ok=True)
r = requests.get(URL, stream=True)
with open(OUT, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

print("Downloaded sample to", OUT)
