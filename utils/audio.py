from pydub import AudioSegment
import numpy as np
import librosa
import io

def load_and_preprocess(audio_bytes):
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    audio = audio.set_channels(1)

    y = np.array(audio.get_array_of_samples()).astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    sr = audio.frame_rate
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    y, _ = librosa.effects.trim(y, top_db=25)
    return y, sr