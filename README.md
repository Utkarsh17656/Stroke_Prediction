# StrokeRisk AI 2.0 - Project Instructions

This project has been updated to support one-click execution and fixed paths after being moved to a new directory.

## One-Click Scripts

- **Run_Stroke_AI.bat**: Start the web server and the AI model. 
  - Access the application at: `http://127.0.0.1:8000`
- **Train_Model.bat**: Re-run the data pipeline to merge datasets and retrain the model.
  - Use this if you update any of the CSV data files.

## Project Structure

- `main.py`: FastAPI web server logic.
- `app.py`: Data processing and model training pipeline.
- `requirements.txt`: List of Python dependencies.
- `venv/`: Local virtual environment containing all necessary packages.
- `static/` & `templates/`: Frontend assets and HTML.
- `*.joblib`: Serialized AI model, scaler, and feature list.

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
