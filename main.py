from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
import os
import traceback

app = FastAPI(title="StrokeRisk AI 2.0")

# Setup paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load model artefacts (prefer calibrated v3 bundle)
MODEL_BUNDLE_V3_PATH = os.path.join(BASE_DIR, "stroke_model_bundle_v3.joblib")
MODEL_V2_PATH = os.path.join(BASE_DIR, "stroke_model_v2.joblib")
SCALER_V2_PATH = os.path.join(BASE_DIR, "scaler_v2.joblib")
FEATURES_V2_PATH = os.path.join(BASE_DIR, "features_v2.joblib")

model = None
threshold = 0.5
expected_features = None
model_version = "unknown"
legacy_scaler = None

if os.path.exists(MODEL_BUNDLE_V3_PATH):
    bundle = joblib.load(MODEL_BUNDLE_V3_PATH)
    model = bundle.get("model")
    threshold = float(bundle.get("threshold", 0.5))
    expected_features = bundle.get("features")
    model_version = bundle.get("model_version", "3.0-calibrated")
    print(f"Loaded model bundle: {model_version}")
elif os.path.exists(MODEL_V2_PATH):
    model = joblib.load(MODEL_V2_PATH)
    model_version = "2.0-legacy"
    if os.path.exists(SCALER_V2_PATH):
        legacy_scaler = joblib.load(SCALER_V2_PATH)
    if os.path.exists(FEATURES_V2_PATH):
        expected_features = joblib.load(FEATURES_V2_PATH)
    print("Loaded legacy v2 model.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
async def predict_get():
    return RedirectResponse(url="/", status_code=303)

@app.post("/predict")
async def predict(
    request: Request,
    gender: str = Form(...),
    age: float = Form(...),
    hypertension: int = Form(0),
    heart_disease: int = Form(0),
    ever_married: str = Form(...),
    work_type: str = Form(...),
    residence_type: str = Form(...),
    cigs_per_day: float = Form(...),
    weight: float = Form(...),
    height: float = Form(...),
    avg_glucose_level: Optional[float] = Form(None),
    no_test: Optional[bool] = Form(False),
    sleep_hours: Optional[float] = Form(None),
    activity_level: str = Form("Medium"),
    alcohol_drinks_per_week: Optional[float] = Form(0),
    stress_level: str = Form("Medium"),
    # Symptom list
    dizziness: bool = Form(False),
    chest_pain: bool = Form(False),
    breath_shortness: bool = Form(False),
    fatigue: bool = Form(False)
):
    try:
        if model is None:
            return HTMLResponse(content="<h1>AI 2.0 Model Not Trained</h1><p>Please run <code>python app.py</code> after placing the 2025 datasets in this folder.</p>", status_code=500)

        # 1. Calculation Logic
        bmi = weight / ((height / 100) ** 2)
        glucose = avg_glucose_level if (avg_glucose_level and not no_test) else 95.0

        # Mapping for 2.0 features
        data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'residence_type': residence_type,
            'avg_glucose_level': glucose,
            'bmi': bmi,
            'cigsPerDay': cigs_per_day,
            'dizziness': 1 if dizziness else 0,
            'chest_pain': 1 if chest_pain else 0,
            'breath_shortness': 1 if breath_shortness else 0,
            'fatigue': 1 if fatigue else 0,
            'sleep_hours': sleep_hours if sleep_hours is not None else np.nan,
            'activity_level': activity_level,
            'alcohol_drinks_per_week': alcohol_drinks_per_week if alcohol_drinks_per_week is not None else np.nan,
            'stress_level': stress_level,
        }

        # 2. Predict Probability
        input_df = pd.DataFrame([data])

        # 3. Mandatory column alignment
        if expected_features:
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            input_df = input_df[expected_features]

        if legacy_scaler is not None:
            input_for_model = legacy_scaler.transform(input_df)
            prob = model.predict_proba(input_for_model)[0][1]
        else:
            prob = model.predict_proba(input_df)[0][1]

        risk = "Critical" if prob >= 0.75 else "High" if prob >= 0.5 else "Moderate" if prob >= threshold else "Low"
        binary_prediction = int(prob >= threshold)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "probability": round(prob * 100, 2),
            "risk": risk,
            "decision_threshold": round(threshold * 100, 1),
            "prediction_label": "Higher Concern" if binary_prediction == 1 else "Lower Concern",
            "model_version": model_version,
            "patient_info": {
                "age": age,
                "bmi": round(bmi, 1),
                "cigs": cigs_per_day,
                "risk_desc": "Probability is calibrated and threshold-tuned to improve consistency."
            }
        })
    except Exception as e:
        traceback.print_exc()
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    