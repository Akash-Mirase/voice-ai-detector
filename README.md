#  AI-Generated Voice Detection API

##  Problem Statement
Design and deploy a REST API that detects whether a given voice sample is AI-generated or spoken by a real human. The API must support multiple languages, accept MP3 audio input, and return a structured JSON response with a confidence score.

##  Solution Overview
This project implements a machine-learning–based voice detection API. The system analyzes acoustic characteristics of speech and classifies the input as AI_GENERATED or HUMAN.

Key Highlights:
- Fully API-based solution
- Uses a trained ML model (no hard-coded rules)
- Supports Base64-encoded MP3 audio
- Returns confidence score (0.0–1.0)
- Includes basic explainability
- Secure via API key authentication

##  Supported Languages
- Tamil
- English
- Hindi
- Malayalam
- Telugu

The model is language-agnostic and operates on audio signal features.

##  Technical Approach

Audio Processing:
- Audio decoding and preprocessing
- Resampling and normalization
- Silence trimming

Feature Extraction:
- MFCC (Mel Frequency Cepstral Coefficients)
- Spectral features
- Temporal speech characteristics

Machine Learning:
- Model: Random Forest Classifier
- Framework: scikit-learn
- Training Data:
  - Real human voice samples
  - AI-generated voice samples (TTS)
- Output: Class label with probability (confidence score)

##  Authentication
The API is protected using an API key.  
The API key must be provided in the request header:

x-api-key

##  API Specification

Endpoint:
POST /api/voice-detection

Request Headers:
x-api-key: <YOUR_API_KEY>  
Content-Type: application/json

Request Body:
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_ENCODED_MP3_AUDIO>"
}

Response Body:
{
  "status": "success",
  "language": "Tamil",
  "classification": "HUMAN",
  "confidenceScore": 0.84,
  "explanation": "Natural speech variability and human-like prosody detected"
}

##  Testing & Validation
Swagger UI (/docs) is provided only for testing and documentation. Swagger is not the final submission. The deployed API endpoint is used by the hackathon Endpoint Tester and evaluation system.

##  Technology Stack
Backend API: FastAPI  
ML Framework: scikit-learn  
Audio Processing: librosa, numpy  
Model Serialization: joblib  
Deployment: Render  
Programming Language: Python  

##  Project Structure
voice-ai-detector/
├── app.py                  # FastAPI application
├── train_model.py          # ML model training
├── requirements.txt
├── data/
│   ├── human/              # Human voice samples
│   └── ai/                 # AI-generated voice samples
├── model/
│   └── detector.pkl        # Trained ML model
├── utils/
│   ├── __init__.py
│   ├── audio.py            # Audio preprocessing
│   └── features.py         # Feature extraction
└── README.md

##  Compliance with Hackathon Rules
- No hard-coded logic
- No external AI-detection APIs
- ML-based classification
- API key authentication
- Proper JSON response structure
- Stateless API (no database required)

##  Conclusion
This project delivers a robust, scalable, and compliant solution for detecting AI-generated voice samples. It strictly follows the problem requirements and is ready for hackathon evaluation and deployment.

##  Author
Akash Mirase , Varadraj Patil
Second-Year Computer Engineering Students 
VIT Pune
