import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def run_2_0_pipeline():
    # Paths relative to this file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Files provided by the user
    framingham_file = os.path.join(BASE_DIR, 'Framingham Dataset.csv')
    symptoms_file = os.path.join(BASE_DIR, 'stroke_risk_dataset.csv')
    synthetic_file = os.path.join(BASE_DIR, 'synthetic_stroke_data.csv')

    # 1. Load Datasets
    print("📂 Loading datasets...")
    df_f = pd.read_csv(framingham_file) if os.path.exists(framingham_file) else None
    df_s = pd.read_csv(symptoms_file) if os.path.exists(symptoms_file) else None
    df_main = pd.read_csv(synthetic_file) if os.path.exists(synthetic_file) else None

    if df_f is None and df_s is None and df_main is None:
        print("❌ Error: No datasets found. Please ensure the CSV files are in the same folder as app.py.")
        return

    # 2. Harmonize & Map Features
    processed_dfs = []

    # --- Process Synthetic 2025 (The Large Base) ---
    if df_main is not None:
        print(f"📊 Processing Synthetic 2025 ({len(df_main)} records)...")
        m = df_main.copy()
        # Initial mapping
        m['gender'] = m['gender'].map({'Male': 1, 'Female': 0, 'Other': 0}).fillna(0)
        m['ever_married'] = m['ever_married'].map({'Yes': 1, 'No': 0}).fillna(0)
        m['Residence_type'] = m['Residence_type'].map({'Urban': 1, 'Rural': 0}).fillna(0)
        # Dummy work type
        m = pd.get_dummies(m, columns=['work_type'])
        
        # Approximate cigsPerDay based on status for the base set
        status_map = {'smokes': 15, 'formerly smoked': 5, 'never smoked': 0, 'Unknown': 0}
        m['cigsPerDay'] = m['smoking_status'].map(status_map).fillna(0)
        
        # Ensure all target columns exist (initially 0 for symptoms)
        for sym in ['dizziness', 'chest_pain', 'breath_shortness', 'fatigue']:
            m[sym] = 0
            
        # Select final 2.0 features
        cols = ['age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 
                'avg_glucose_level', 'bmi', 'gender', 'cigsPerDay',
                'dizziness', 'chest_pain', 'breath_shortness', 'fatigue',
                'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 
                'work_type_Self-employed', 'work_type_children', 'stroke']
        
        # Missing columns check
        for c in cols:
            if c not in m.columns: m[c] = 0
            
        processed_dfs.append(m[cols])

    # --- Process Framingham (Clinical Precision) ---
    if df_f is not None:
        print(f"🏥 Processing Framingham Study ({len(df_f)} records)...")
        f = df_f.copy()
        f.rename(columns={'AGE': 'age', 'CIGPDAY': 'cigsPerDay', 'BMI': 'bmi', 'GLUCOSE': 'avg_glucose_level', 'HYPERTEN': 'hypertension', 'SEX': 'gender', 'STROKE': 'stroke'}, inplace=True)
        f['gender'] = f['gender'].map({1: 1, 2: 0}).fillna(0) # Framingham Coding
        f['heart_disease'] = f['PREVCHD'] # Approximation
        f['ever_married'] = 1 # Not in Framingham, assuming 1 for weight balance
        f['Residence_type'] = 1 # Assuming Urban
        
        # Dummy columns (Empty for Framingham)
        for col in ['work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'dizziness', 'chest_pain', 'breath_shortness', 'fatigue']:
            f[col] = 0
            
        processed_dfs.append(f[cols])

    # --- Process Symptoms (Diagnostic Intelligence) ---
    if df_s is not None:
        print(f"🧠 Processing Symptom-based Data ({len(df_s)} records)...")
        s = df_s.copy()
        s.rename(columns={'Age': 'age', 'At Risk (Binary)': 'stroke', 'Dizziness': 'dizziness', 'Chest Pain': 'chest_pain', 'Shortness of Breath': 'breath_shortness', 'Fatigue & Weakness': 'fatigue'}, inplace=True)
        # Add missing columns
        for c in cols:
            if c not in s.columns: s[c] = 0
        processed_dfs.append(s[cols])

    # 3. Concatenate all data
    print("🔗 Merging all sources into AI 2.0 Unified Dataset...")
    final_df = pd.concat(processed_dfs, ignore_index=True)
    final_df = final_df.dropna()
    print(f"✅ Final training pool: {len(final_df)} records.")

    # 4. Train-Test Split
    X = final_df.drop('stroke', axis=1)
    y = final_df['stroke']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # 5. Handle Imbalance with SMOTE
    print("⚖️ Balancing clinical classes...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. Scaling
    print("📏 Scaling lifestyle markers...")
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # 7. Model Training (Optimized LightGBM)
    print("🏗️ Building LightGBM 2.0 Model...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train_res, y_train_res)

    # 8. Evaluation & Save
    from sklearn.metrics import roc_auc_score
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    print(f"✨ AI 2.0 ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    print("💾 Saving AI 2.0 Artefacts...")
    joblib.dump(model, os.path.join(BASE_DIR, 'stroke_model_v2.joblib'))
    joblib.dump(scaler, os.path.join(BASE_DIR, 'scaler_v2.joblib'))
    # Save the feature names order for the API
    joblib.dump(list(X.columns), os.path.join(BASE_DIR, 'features_v2.joblib'))
    print("🎯 AI 2.0 Pipeline Integrated Successfully.")

if __name__ == "__main__":
    run_2_0_pipeline()
