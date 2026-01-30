from pydub import AudioSegment
import numpy as np
import librosa
import io


def load_and_preprocess(audio_bytes):
    audio_io = io.BytesIO(audio_bytes)

    MAX_DURATION = 10  # seconds (IMPORTANT)

    y, sr = librosa.load(audio_io, sr=16000, mono=True, duration=MAX_DURATION)

    return y, sr
