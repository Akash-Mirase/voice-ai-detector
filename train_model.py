import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from utils.audio import load_and_preprocess
from utils.features import extract_features

X, y = [], []

for label, folder in [(0, "data/human"), (1, "data/ai")]:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            with open(path, "rb") as f:
                audio_bytes = f.read()

            y_audio, sr = load_and_preprocess(audio_bytes)

            if len(y_audio) < sr * 0.5:
                continue

            features = extract_features(y_audio, sr)
            X.append(features)
            y.append(label)

        except:
            continue

if len(X) == 0:
    raise RuntimeError("No valid audio files found")

model = RandomForestClassifier(n_estimators=100, random_state=42)
# SAFETY CHECK 
unique_classes = set(y)
if len(unique_classes) < 2:
    raise RuntimeError(
        f"Training failed: Only one class found: {unique_classes}. "
        "Please add data for both HUMAN and AI."
    )

model.fit(X, y)


joblib.dump(model, "model/detector.pkl")
print("✅ Model trained successfully")
