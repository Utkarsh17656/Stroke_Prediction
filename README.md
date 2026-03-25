# StrokeRisk AI 3.0 - Project Instructions

This project supports one-click execution with an upgraded calibrated ML pipeline for more stable and realistic risk probabilities.

## One-Click Scripts

- **Run_Stroke_AI.bat**: Start the web server using the latest trained model artefacts.
  - Access the application at: `http://127.0.0.1:8000`
- **Train_Model.bat**: Re-run the data pipeline and retrain the model.
  - You only need to run this when your CSV datasets change or when you intentionally want to rebuild the model.
  - You do **not** need to run this every time before starting the web app.

## Project Structure

- `main.py`: FastAPI web server and inference logic.
- `app.py`: Data processing and calibrated model training pipeline.
- `requirements.txt`: List of Python dependencies.
- `venv/`: Local virtual environment containing all necessary packages.
- `static/` & `templates/`: Frontend assets and HTML.
- `stroke_model_bundle_v3.joblib`: Serialized calibrated model + threshold + feature metadata.
- Legacy files (`stroke_model_v2.joblib`, `scaler_v2.joblib`, `features_v2.joblib`) are kept for backward compatibility.

## Model Notes

- Current default model: calibrated gradient boosting classifier.
- Why it is better: handles missing fields safely, avoids hard-coded fake values for unavailable columns, and uses validation-tuned thresholding for better practical decision consistency.

## Docker Execution (Containerized)

If you have Docker installed, you can run the entire project in a single isolated container.

### Option 1: Using Docker Compose (Recommended)
1. Open a terminal in the project folder.
2. Run: `docker-compose up --build`
3. Access at: `http://localhost:8000`

### Option 2: Using Standard Docker Commands
1. **Build the image**:
   `docker build -t stroke-ai .`
2. **Run the container**:
   `docker run -p 8000:8000 stroke-ai`

## Troubleshooting

If you encounter issues, ensure you are running the project from the `Stroke_2.0` folder. The batch files automatically handle virtual environment activation for you.
