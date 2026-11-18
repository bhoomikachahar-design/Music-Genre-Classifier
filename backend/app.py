"""
FastAPI Backend Application for Music Genre Classifier & Explorer
GLA University - ML Software Project
"""

import os
import io
import json
import uuid
import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from core.audio_processor import AudioProcessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = FastAPI(
    title="Music Genre Classifier & Explorer API",
    description="End-to-End Audio Signal Processing and Multiclass Machine Learning Genre Classification",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Genre Musical Profile Descriptions (Educational context)
GENRE_PROFILES = {
    "blues": {
        "title": "Blues",
        "description": "Rooted in African-American musical traditions, characterized by 12-bar chord progressions, expressive pitch bending (blue notes), soulful guitar licks, and emotional vocal phrasings.",
        "key_features": "Prominent mid-frequency warmth, moderate tempo (70-110 BPM), dynamic vocal/guitar interplay.",
        "typical_instruments": ["Acoustic & Electric Guitar", "Harmonica", "Piano", "Bass"]
    },
    "classical": {
        "title": "Classical",
        "description": "Acoustic orchestral compositions characterized by high dynamic range, intricate harmonic counterpoint, sophisticated orchestration, and minimal percussive distortion.",
        "key_features": "High spectral dynamic range, low percussive noise, rich harmonic STFT content, zero-crossing rates varying with phrasing.",
        "typical_instruments": ["Violin", "Cello", "Grand Piano", "Woodwinds", "Brass"]
    },
    "country": {
        "title": "Country",
        "description": "American roots music emphasizing acoustic stringed instruments, storytelling vocal delivery, distinct melodic twang, and steady rhythmic strumming.",
        "key_features": "Acoustic string timbres (MFCCs 2-6 prominent), balanced spectral centroid, steady mid-tempo.",
        "typical_instruments": ["Acoustic Guitar", "Banjo", "Fiddle", "Pedal Steel", "Drums"]
    },
    "disco": {
        "title": "Disco",
        "description": "Dance music with an infectious 'four-on-the-floor' beat, synchronized off-beat open hi-hats, melodic electric basslines, and brass/string stabs.",
        "key_features": "Steady tempo (~120-130 BPM), high percussive energy, sharp high-frequency rolloff from hi-hats.",
        "typical_instruments": ["Four-on-the-Floor Kick", "Funky Bass", "Electric Rhythm Guitar", "Synthesizer"]
    },
    "hiphop": {
        "title": "Hip-Hop",
        "description": "Urban music genre built around rhythmic vocal delivery (rapping), prominent sub-bass 808 hits, sampled loops, and snappy syncopated snares.",
        "key_features": "Heavy low-frequency energy (MFCC 1 & 2), distinct percussive transient peaks, 80-100 BPM tempo range.",
        "typical_instruments": ["808 Sub-Bass", "Drum Machine Snares/Hi-Hats", "Samplers", "Turntables"]
    },
    "jazz": {
        "title": "Jazz",
        "description": "Complex harmonies featuring 7th, 9th and altered chords, swing polyrhythms, syncopated ride cymbal patterns, and virtuosic improvisation.",
        "key_features": "High harmonic variance, polyphonic chroma spread, dynamic swing beats, warm acoustic timbres.",
        "typical_instruments": ["Saxophone", "Trumpet", "Upright Double Bass", "Piano", "Brushed Drums"]
    },
    "metal": {
        "title": "Metal",
        "description": "High-intensity genre driven by heavy guitar distortion, fast palm-muted riffs, aggressive double-bass drumming, and powerful vocal styles.",
        "key_features": "Extreme high-frequency saturation, very high spectral centroid and bandwidth, intense continuous RMS energy.",
        "typical_instruments": ["Distorted Electric Guitars", "Double-Kick Drums", "Bass Guitar", "Screaming Vocals"]
    },
    "pop": {
        "title": "Pop",
        "description": "Mainstream, hook-driven music with clean production, verse-chorus song structures, catchy melodies, and polished electronic percussion.",
        "key_features": "Balanced spectral balance, modern loudness compression, high harmonic clarity, upbeat tempo (110-128 BPM).",
        "typical_instruments": ["Vocal Lead", "Synthesizers", "Electronic Beats", "Polished Bass"]
    },
    "reggae": {
        "title": "Reggae",
        "description": "Jamaican genre defined by accented off-beat chord stabs (skank), deep melodic basslines, relaxed half-time groove, and lyrical social consciousness.",
        "key_features": "Distinctive off-beat rhythm spikes, strong low-end bass presence, relaxed tempo (60-90 BPM).",
        "typical_instruments": ["Bass Guitar", "Off-beat Organ/Guitar Skank", "Drums", "Percussion"]
    },
    "rock": {
        "title": "Rock",
        "description": "Guitar-centric genre centered around an assertive 4/4 backbeat (snare on beats 2 and 4), amplified power chords, energetic vocals, and melodic solos.",
        "key_features": "Balanced mid-to-high spectral rolloff, energetic percussive transients, prominent harmonic distortion.",
        "typical_instruments": ["Electric Guitar", "Acoustic Drum Kit", "Electric Bass", "Vocals"]
    }
}

# Load ML Models and Artifacts
models = {}
scaler = None
label_encoder = None
feature_names = []
model_metrics = {}

def load_ml_artifacts():
    global models, scaler, label_encoder, feature_names, model_metrics
    try:
        rf_path = os.path.join(MODELS_DIR, "random_forest.joblib")
        svm_path = os.path.join(MODELS_DIR, "svm.joblib")
        et_path = os.path.join(MODELS_DIR, "extra_trees.joblib")
        scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
        le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        fn_path = os.path.join(MODELS_DIR, "feature_names.json")
        met_path = os.path.join(MODELS_DIR, "metrics.json")

        if os.path.exists(rf_path):
            models["random_forest"] = joblib.load(rf_path)
        if os.path.exists(svm_path):
            models["svm"] = joblib.load(svm_path)
        if os.path.exists(et_path):
            models["extra_trees"] = joblib.load(et_path)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        if os.path.exists(le_path):
            label_encoder = joblib.load(le_path)
        if os.path.exists(fn_path):
            with open(fn_path, "r", encoding="utf-8") as f:
                feature_names = json.load(f)
        if os.path.exists(met_path):
            with open(met_path, "r", encoding="utf-8") as f:
                model_metrics = json.load(f)
        print("ML artifacts successfully loaded into backend memory.")
    except Exception as e:
        print(f"Warning loading ML artifacts: {e}")

load_ml_artifacts()
audio_processor = AudioProcessor()

# Prediction History Store
def load_history() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(entry: Dict[str, Any]):
    history = load_history()
    history.insert(0, entry)
    # keep last 50
    history = history[:50]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def predict_from_features(feats_dict: Dict[str, float], model_choice: str = "ensemble"):
    """
    Given an extracted feature dict, run inference with chosen model.
    """
    if not models or label_encoder is None:
        raise HTTPException(status_code=500, detail="Models are not initialized. Please train models first.")

    # Align features with feature_names
    row = [feats_dict.get(fn, 0.0) for fn in feature_names]
    X_sample = pd.DataFrame([row], columns=feature_names)

    class_names = list(label_encoder.classes_)

    if model_choice == "random_forest" and "random_forest" in models:
        probs = models["random_forest"].predict_proba(X_sample)[0]
    elif model_choice == "svm" and "svm" in models:
        X_scaled = scaler.transform(X_sample)
        probs = models["svm"].predict_proba(X_scaled)[0]
    else:
        # Default: Ensemble blend (40% RF + 35% SVM + 25% ExtraTrees)
        rf_p = models["random_forest"].predict_proba(X_sample)[0] if "random_forest" in models else np.zeros(len(class_names))
        et_p = models["extra_trees"].predict_proba(X_sample)[0] if "extra_trees" in models else np.zeros(len(class_names))
        if "svm" in models and scaler is not None:
            svm_p = models["svm"].predict_proba(scaler.transform(X_sample))[0]
        else:
            svm_p = rf_p
        probs = (rf_p * 0.40) + (svm_p * 0.35) + (et_p * 0.25)

    pred_idx = int(np.argmax(probs))
    pred_genre = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    # Sorted probabilities for all 10 genres
    prob_list = []
    for i, c in enumerate(class_names):
        prob_list.append({
            "genre": c,
            "probability": round(float(probs[i]), 4),
            "percentage": round(float(probs[i]) * 100, 2)
        })
    prob_list.sort(key=lambda x: x["probability"], reverse=True)

    profile = GENRE_PROFILES.get(pred_genre, {
        "title": pred_genre.capitalize(),
        "description": "Music genre identified by distinctive spectral timbre and rhythm.",
        "key_features": "Harmonic and percussive profile matched by classifier.",
        "typical_instruments": ["Various"]
    })

    return {
        "predicted_genre": pred_genre,
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100, 2),
        "model_used": model_choice,
        "probabilities": prob_list,
        "profile": profile
    }

# ----------------- API ENDPOINTS -----------------

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Music Genre Classifier & Explorer API",
        "models_loaded": list(models.keys()),
        "classes": list(label_encoder.classes_) if label_encoder else []
    }

@app.post("/api/predict")
async def predict_audio(
    file: UploadFile = File(...),
    model: str = Query("ensemble", pattern="^(ensemble|random_forest|svm)$")
):
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Save audio file to uploads for playback
        ext = os.path.splitext(file.filename)[1].lower() or ".wav"
        unique_name = f"{uuid.uuid4().hex[:10]}{ext}"
        save_path = os.path.join(UPLOADS_DIR, unique_name)
        with open(save_path, "wb") as f:
            f.write(content)

        # Process audio
        y, sr = audio_processor.load_audio(content)
        features, mfccs = audio_processor.extract_gtzan_features(y, sr)
        vis_data = audio_processor.generate_visualizations(y, sr, mfccs)

        # Run prediction
        prediction = predict_from_features(features, model_choice=model)

        # Record History
        history_entry = {
            "id": unique_name,
            "filename": file.filename,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_genre": prediction["predicted_genre"],
            "confidence_percentage": prediction["confidence_percentage"],
            "model_used": model,
            "audio_url": f"/api/audio/{unique_name}",
            "duration": vis_data["summary"]["duration"],
            "tempo_bpm": vis_data["summary"]["estimated_tempo_bpm"]
        }
        save_history(history_entry)

        return {
            "status": "success",
            "file_info": {
                "original_name": file.filename,
                "audio_url": f"/api/audio/{unique_name}",
                "file_id": unique_name
            },
            "prediction": prediction,
            "visualizations": vis_data,
            "extracted_features": features
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/predict-sample/{sample_id}")
def predict_sample(
    sample_id: str,
    model: str = Query("ensemble", pattern="^(ensemble|random_forest|svm)$")
):
    sample_filename = f"sample_{sample_id.lower()}.wav"
    sample_path = os.path.join(SAMPLES_DIR, sample_filename)

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")

    try:
        y, sr = audio_processor.load_audio(sample_path)
        features, mfccs = audio_processor.extract_gtzan_features(y, sr)
        vis_data = audio_processor.generate_visualizations(y, sr, mfccs)

        prediction = predict_from_features(features, model_choice=model)

        # Record History
        history_entry = {
            "id": sample_filename,
            "filename": f"Sample: {sample_id.capitalize()}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "predicted_genre": prediction["predicted_genre"],
            "confidence_percentage": prediction["confidence_percentage"],
            "model_used": model,
            "audio_url": f"/api/audio/sample/{sample_filename}",
            "duration": vis_data["summary"]["duration"],
            "tempo_bpm": vis_data["summary"]["estimated_tempo_bpm"]
        }
        save_history(history_entry)

        return {
            "status": "success",
            "file_info": {
                "original_name": f"Demo Sample: {sample_id.capitalize()}",
                "audio_url": f"/api/audio/sample/{sample_filename}",
                "file_id": sample_filename
            },
            "prediction": prediction,
            "visualizations": vis_data,
            "extracted_features": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating sample: {str(e)}")

@app.get("/api/samples")
def list_samples():
    samples_info = [
        {
            "id": "classical",
            "title": "Classical Piano Arpeggio",
            "description": "Harmonic acoustic arpeggios with piano resonance.",
            "audio_url": "/api/audio/sample/sample_classical.wav"
        },
        {
            "id": "rock",
            "title": "Rock Power Chords & Drums",
            "description": "Overdriven electric guitar riffs with punchy kick/snare.",
            "audio_url": "/api/audio/sample/sample_rock.wav"
        },
        {
            "id": "jazz",
            "title": "Jazz Swing & 7th Chords",
            "description": "Sophisticated swing rhythm with ride cymbal and polyphonic chords.",
            "audio_url": "/api/audio/sample/sample_jazz.wav"
        },
        {
            "id": "disco",
            "title": "Disco Four-on-the-Floor",
            "description": "124 BPM dance beat with off-beat hi-hats and pulsing bassline.",
            "audio_url": "/api/audio/sample/sample_disco.wav"
        },
        {
            "id": "metal",
            "title": "Metal High-Gain Chug",
            "description": "Fast double-kick drums with distorted low-end harmonics.",
            "audio_url": "/api/audio/sample/sample_metal.wav"
        }
    ]
    return samples_info

@app.get("/api/audio/{filename}")
def get_uploaded_audio(filename: str):
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/api/audio/sample/{filename}")
def get_sample_audio(filename: str):
    file_path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample audio not found")
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/api/models")
def get_models_info():
    if not model_metrics:
        return {"error": "Model metrics not available. Please train models."}
    return model_metrics

@app.get("/api/history")
def get_history():
    return load_history()

@app.get("/api/history/export")
def export_history(format: str = Query("json", pattern="^(json|csv)$")):
    history = load_history()
    if format == "csv":
        df = pd.DataFrame(history)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=classification_history.csv"
        return response
    else:
        return JSONResponse(content=history)

@app.get("/api/pipeline")
def get_pipeline_info():
    return {
        "title": "End-to-End Audio Classification Pipeline",
        "stages": [
            {
                "step": 1,
                "name": "Audio Ingestion & Resampling",
                "description": "Uploaded audio (WAV, MP3, etc.) is decoded, converted to mono, normalized for peak amplitude, and resampled to the benchmark 22,050 Hz sampling frequency."
            },
            {
                "step": 2,
                "name": "Short-Time Fourier Transform (STFT)",
                "description": "The time-domain waveform is converted into time-frequency representation via Short-Time Fourier Transform with standard window size."
            },
            {
                "step": 3,
                "name": "Acoustic Feature Extraction",
                "description": "Extracts 20 Mel-frequency cepstral coefficients (MFCCs mean & var), Chroma STFT, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Zero Crossing Rate, Harmony/Percussion components, and Tempo."
            },
            {
                "step": 4,
                "name": "Feature Vector Assembly",
                "description": "The aggregated acoustic descriptors are consolidated into a calibrated 58-dimensional numerical feature vector matching the GTZAN feature format."
            },
            {
                "step": 5,
                "name": "Supervised Multiclass Classification",
                "description": "The feature vector is evaluated by trained supervised classifiers (Random Forest, SVM, or Ensemble Soft-Voting) to compute calibrated posterior probability distributions across 10 music genres."
            }
        ]
    }

# Mount static frontend
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
