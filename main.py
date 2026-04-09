from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
import os
import traceback

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(title="StrokeRisk AI 2.0")
logger = logging.getLogger("stroke_ai")
logging.basicConfig(level=logging.INFO)

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

ALLOWED_GENDERS = {"Male", "Female"}
ALLOWED_WORK_TYPES = {"Private", "Govt_job", "Self-employed", "children"}
ALLOWED_RESIDENCE_TYPES = {"Urban", "Rural"}
ALLOWED_ACTIVITY = {"Low", "Medium", "High"}
ALLOWED_STRESS = {"Low", "Medium", "High"}

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


class StrokePredictRequest(BaseModel):
    gender: str
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(0, ge=0, le=1)
    heart_disease: int = Field(0, ge=0, le=1)
    ever_married: str
    work_type: str
    residence_type: str
    cigs_per_day: float = Field(0, ge=0, le=120)
    weight: float = Field(..., gt=0)
    height: float = Field(..., gt=0)
    avg_glucose_level: Optional[float] = None
    no_test: bool = False
    sleep_hours: Optional[float] = None
    activity_level: str = "Medium"
    alcohol_drinks_per_week: Optional[float] = 0
    stress_level: str = "Medium"
    dizziness: bool = False
    chest_pain: bool = False
    breath_shortness: bool = False
    fatigue: bool = False


def _validate_payload(payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if payload["gender"] not in ALLOWED_GENDERS:
        errors.append("Gender must be Male or Female.")
    if payload["work_type"] not in ALLOWED_WORK_TYPES:
        errors.append("Work type is invalid.")
    if payload["residence_type"] not in ALLOWED_RESIDENCE_TYPES:
        errors.append("Residence type is invalid.")
    if payload["activity_level"] not in ALLOWED_ACTIVITY:
        errors.append("Activity level must be Low, Medium, or High.")
    if payload["stress_level"] not in ALLOWED_STRESS:
        errors.append("Stress level must be Low, Medium, or High.")

    age = payload["age"]
    height = payload["height"]
    weight = payload["weight"]
    cigs = payload["cigs_per_day"]

    if age < 0 or age > 120:
        errors.append("Age must be between 0 and 120 years.")
    if height < 100 or height > 250:
        errors.append("Height must be between 100 and 250 cm.")
    if weight < 25 or weight > 350:
        errors.append("Weight must be between 25 and 350 kg.")
    if cigs < 0 or cigs > 120:
        errors.append("Cigarettes per day must be between 0 and 120.")

    glucose = payload.get("avg_glucose_level")
    if glucose is not None and (glucose < 40 or glucose > 600):
        errors.append("Glucose must be between 40 and 600 mg/dL.")

    sleep_hours = payload.get("sleep_hours")
    if sleep_hours is not None and (sleep_hours < 0 or sleep_hours > 24):
        errors.append("Sleep hours must be between 0 and 24.")

    alcohol = payload.get("alcohol_drinks_per_week")
    if alcohol is not None and (alcohol < 0 or alcohol > 80):
        errors.append("Alcohol drinks per week must be between 0 and 80.")

    return errors


def _clinical_overlay(payload: Dict[str, Any], bmi: float, glucose_used: float) -> Dict[str, Any]:
    score = 0
    factors: List[str] = []
    recommendations: List[str] = []
    urgent_flags: List[str] = []

    age = payload["age"]
    if age >= 75:
        score += 4
        factors.append("Age >= 75")
    elif age >= 65:
        score += 3
        factors.append("Age 65-74")
    elif age >= 55:
        score += 2
        factors.append("Age 55-64")
    elif age >= 45:
        score += 1
        factors.append("Age 45-54")

    if payload["hypertension"] == 1:
        score += 4
        factors.append("Hypertension")
        recommendations.append("Maintain strict blood pressure control and regular follow-ups.")

    if payload["heart_disease"] == 1:
        score += 3
        factors.append("Cardiovascular disease history")
        recommendations.append("Coordinate care with cardiology for stroke-risk reduction.")

    if bmi >= 35:
        score += 2
        factors.append("BMI >= 35")
        recommendations.append("Consider supervised weight reduction plan.")
    elif bmi >= 30:
        score += 1
        factors.append("BMI 30-34.9")

    if glucose_used >= 200:
        score += 3
        factors.append("Glucose >= 200 mg/dL")
        recommendations.append("Urgent diabetes evaluation and glucose management is advised.")
    elif glucose_used >= 126:
        score += 2
        factors.append("Glucose 126-199 mg/dL")
        recommendations.append("Check HbA1c and fasting glucose confirmation.")
    elif glucose_used >= 100:
        score += 1
        factors.append("Glucose 100-125 mg/dL")

    cigs = payload["cigs_per_day"]
    if cigs >= 20:
        score += 3
        factors.append("Heavy smoking")
        recommendations.append("Immediate smoking cessation support is recommended.")
    elif cigs >= 1:
        score += 1
        factors.append("Current smoking")

    activity_level = payload["activity_level"]
    if activity_level == "Low":
        score += 2
        factors.append("Low physical activity")
        recommendations.append("Target at least 150 min/week moderate activity if medically appropriate.")
    elif activity_level == "Medium":
        score += 1

    alcohol = payload["alcohol_drinks_per_week"] or 0
    if alcohol >= 15:
        score += 2
        factors.append("High alcohol intake")
        recommendations.append("Reduce alcohol use and monitor liver and metabolic profile.")
    elif alcohol >= 8:
        score += 1

    stress = payload["stress_level"]
    if stress == "High":
        score += 2
        factors.append("High stress")
    elif stress == "Medium":
        score += 1

    symptoms_count = sum(
        int(payload[name]) for name in ["dizziness", "chest_pain", "breath_shortness", "fatigue"]
    )
    if symptoms_count >= 1:
        score += 2
        factors.append(f"{symptoms_count} concerning symptom(s)")
    if symptoms_count >= 2:
        score += 2

    if payload["chest_pain"]:
        urgent_flags.append("Chest pain reported")
    if payload["breath_shortness"]:
        urgent_flags.append("Shortness of breath reported")
    if payload["dizziness"] and age >= 55:
        urgent_flags.append("Dizziness in higher age group")

    if score >= 17:
        clinical_band = "Critical"
    elif score >= 11:
        clinical_band = "High"
    elif score >= 6:
        clinical_band = "Moderate"
    else:
        clinical_band = "Low"

    recommendations.append("This tool is decision support only and does not replace physician diagnosis.")

    return {
        "clinical_score": score,
        "clinical_band": clinical_band,
        "factors": factors,
        "urgent_flags": urgent_flags,
        "recommendations": list(dict.fromkeys(recommendations)),
    }


def _predict_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if model is None:
        raise RuntimeError("Model is not available. Train or restore model artifacts first.")

    bmi = payload["weight"] / ((payload["height"] / 100) ** 2)
    glucose = payload["avg_glucose_level"] if (payload.get("avg_glucose_level") is not None and not payload["no_test"]) else 95.0

    features = {
        "gender": payload["gender"],
        "age": payload["age"],
        "hypertension": payload["hypertension"],
        "heart_disease": payload["heart_disease"],
        "ever_married": payload["ever_married"],
        "work_type": payload["work_type"],
        "residence_type": payload["residence_type"],
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "cigsPerDay": payload["cigs_per_day"],
        "dizziness": 1 if payload["dizziness"] else 0,
        "chest_pain": 1 if payload["chest_pain"] else 0,
        "breath_shortness": 1 if payload["breath_shortness"] else 0,
        "fatigue": 1 if payload["fatigue"] else 0,
        "sleep_hours": payload["sleep_hours"] if payload["sleep_hours"] is not None else np.nan,
        "activity_level": payload["activity_level"],
        "alcohol_drinks_per_week": payload["alcohol_drinks_per_week"] if payload["alcohol_drinks_per_week"] is not None else np.nan,
        "stress_level": payload["stress_level"],
    }

    input_df = pd.DataFrame([features])
    if expected_features:
        for col in expected_features:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[expected_features]

    if legacy_scaler is not None:
        input_for_model = legacy_scaler.transform(input_df)
        prob = float(model.predict_proba(input_for_model)[0][1])
    else:
        prob = float(model.predict_proba(input_df)[0][1])

    ml_band = "Critical" if prob >= 0.75 else "High" if prob >= 0.5 else "Moderate" if prob >= threshold else "Low"
    binary_prediction = int(prob >= threshold)
    overlay = _clinical_overlay(payload=payload, bmi=bmi, glucose_used=glucose)

    return {
        "probability": round(prob * 100, 2),
        "risk": ml_band,
        "decision_threshold": round(threshold * 100, 1),
        "prediction_label": "Higher Concern" if binary_prediction == 1 else "Lower Concern",
        "model_version": model_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_info": {
            "age": payload["age"],
            "bmi": round(bmi, 1),
            "cigs": payload["cigs_per_day"],
            "risk_desc": "ML probability is calibrated and threshold-tuned for consistency.",
        },
        "clinical_overlay": overlay,
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "error_messages": [], "form_data": {}})


@app.get("/healthz")
async def healthz():
    return JSONResponse(
        {
            "status": "ok" if model is not None else "degraded",
            "model_loaded": model is not None,
            "model_version": model_version,
            "threshold": threshold,
        }
    )

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

        payload = {
            "gender": gender,
            "age": float(age),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "ever_married": ever_married,
            "work_type": work_type,
            "residence_type": residence_type,
            "cigs_per_day": float(cigs_per_day),
            "weight": float(weight),
            "height": float(height),
            "avg_glucose_level": float(avg_glucose_level) if avg_glucose_level is not None else None,
            "no_test": bool(no_test),
            "sleep_hours": float(sleep_hours) if sleep_hours is not None else None,
            "activity_level": activity_level,
            "alcohol_drinks_per_week": float(alcohol_drinks_per_week) if alcohol_drinks_per_week is not None else 0.0,
            "stress_level": stress_level,
            "dizziness": bool(dizziness),
            "chest_pain": bool(chest_pain),
            "breath_shortness": bool(breath_shortness),
            "fatigue": bool(fatigue),
        }

        errors = _validate_payload(payload)
        if errors:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error_messages": errors,
                    "form_data": payload,
                },
                status_code=422,
            )

        result = _predict_from_payload(payload)
        return templates.TemplateResponse("result.html", {"request": request, **result})
    except Exception as e:
        logger.exception("Prediction failure")
        traceback.print_exc()
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)


@app.post("/api/v1/predict")
async def predict_api(payload: StrokePredictRequest):
    as_dict = payload.model_dump()
    errors = _validate_payload(as_dict)
    if errors:
        return JSONResponse({"errors": errors}, status_code=422)

    try:
        result = _predict_from_payload(as_dict)
        return JSONResponse(result)
    except Exception:
        logger.exception("API prediction failure")
        return JSONResponse({"error": "Prediction service failed."}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    