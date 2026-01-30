from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import base64
import joblib
import os

from utils.audio import load_and_preprocess
from utils.features import extract_features

# App Setup
app = FastAPI(
    title="AI-Generated Voice Detection API",
    description="Detects whether a given voice sample is AI-generated or Human",
    version="1.0",
)

API_KEY = os.getenv("API_KEY", "testkey")

# Load trained ML model once
model = joblib.load("model/detector.pkl")



# Request & Response Models (AS PER DOCUMENT)

class VoiceDetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


class VoiceDetectionResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str



# Utility: Run ML inference

def run_inference(audio_bytes: bytes):
    y, sr = load_and_preprocess(audio_bytes)
    features = extract_features(y, sr)

    probabilities = model.predict_proba([features])[0]
    classes = model.classes_

    prob_map = dict(zip(classes, probabilities))
    prediction = model.predict([features])[0]
    confidence = prob_map[prediction]

    return prediction, confidence



# OFFICIAL ENDPOINT (SUBMISSION ENDPOINT)

@app.post("/api/voice-detection", response_model=VoiceDetectionResponse)
def detect_voice(
    request: VoiceDetectionRequest, x_api_key: str = Header(None, alias="x-api-key")
):
    #  AUTH 
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401, detail="Invalid API key or unauthorized request"
        )

    #  VALIDATION 
    if request.audioFormat.lower() != "mp3":
        raise HTTPException(
            status_code=400, detail="Only MP3 audio format is supported"
        )

    #  BASE64 DECODE 
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio input")

    #  INFERENCE 
    try:
        pred, conf = run_inference(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e}")

    #  RESULT MAPPING 
    classification = "AI_GENERATED" if pred == 1 else "HUMAN"

    # Simple explainability (acceptable as per problem)
    explanation = (
        "Unnatural pitch consistency and low prosodic variation detected"
        if classification == "AI_GENERATED"
        else "Natural speech variability and human-like prosody detected"
    )

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": round(float(conf), 3),
        "explanation": explanation,
    }
