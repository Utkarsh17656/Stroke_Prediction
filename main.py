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

# Load model and scaler (AI 2.0 versions)
MODEL_PATH = os.path.join(BASE_DIR, "stroke_model_v2.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_v2.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "features_v2.joblib")

model = None
scaler = None
expected_features = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    if os.path.exists(FEATURES_PATH):
        expected_features = joblib.load(FEATURES_PATH)
    print("AI 2.0 Model loaded successfully.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.route("/predict", methods=["GET", "POST"])
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
    # Symptom list
    dizziness: bool = Form(False),
    chest_pain: bool = Form(False),
    breath_shortness: bool = Form(False),
    fatigue: bool = Form(False)
):
    # Handle GET requests (likely redirects from POST due to protocol mismatch)
    if request.method == "GET":
        return RedirectResponse(url="/", status_code=303)

    try:
        if not model or not scaler:
            return HTMLResponse(content="<h1>AI 2.0 Model Not Trained</h1><p>Please run <code>python app.py</code> after placing the 2025 datasets in this folder.</p>", status_code=500)

        # 1. Calculation Logic
        bmi = weight / ((height / 100) ** 2)
        glucose = avg_glucose_level if (avg_glucose_level and not no_test) else 95.0

        # Mapping for 2.0 features
        data = {
            'gender': 1 if gender == 'Male' else 0,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': 1 if ever_married == 'Yes' else 0,
            'Residence_type': 1 if residence_type == 'Urban' else 0,
            'avg_glucose_level': glucose,
            'bmi': bmi,
            'cigsPerDay': cigs_per_day,
            'dizziness': 1 if dizziness else 0,
            'chest_pain': 1 if chest_pain else 0,
            'breath_shortness': 1 if breath_shortness else 0,
            'fatigue': 1 if fatigue else 0
        }

        # 2. Add Dummy features for Work Type (One-Hot)
        work_types = ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children']
        for wt in work_types:
            data[f'work_type_{wt}'] = 1 if work_type == wt else 0

        # 3. Predict Probability
        input_df = pd.DataFrame([data])
        
        # 4. Mandatory column alignment (Critical for 2.0 Accuracy)
        if expected_features:
            # Ensure only required columns are present and in exact order
            input_df = input_df[expected_features]
        else:
            # Fallback to current internal order if file missing
            cols = ['age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 
                    'avg_glucose_level', 'bmi', 'gender', 'cigsPerDay',
                    'dizziness', 'chest_pain', 'breath_shortness', 'fatigue',
                    'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 
                    'work_type_Self-employed', 'work_type_children']
            input_df = input_df[cols]

        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]

        risk = "Critical" if prob > 0.7 else "High" if prob > 0.4 else "Moderate" if prob > 0.15 else "Low"

        return templates.TemplateResponse("result.html", {
            "request": request,
            "probability": round(prob * 100, 2),
            "risk": risk,
            "patient_info": {
                "age": age,
                "bmi": round(bmi, 1),
                "cigs": cigs_per_day,
                "risk_desc": "Consistent with high-scale 2025 trends."
            }
        })
    except Exception as e:
        traceback.print_exc()
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    