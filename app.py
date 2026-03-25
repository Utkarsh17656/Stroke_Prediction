import os
import urllib.request
import warnings
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

BRFSS_2022_ZIP_URL = "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip"
BRFSS_2022_ZIP_NAME = "LLCP2022XPT.zip"
BRFSS_2022_XPT_NAME = "LLCP2022.XPT"
BRFSS_MAX_ROWS = 100000


FEATURE_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "residence_type",
    "avg_glucose_level",
    "bmi",
    "cigsPerDay",
    "dizziness",
    "chest_pain",
    "breath_shortness",
    "fatigue",
    "sleep_hours",
    "activity_level",
    "alcohol_drinks_per_week",
    "stress_level",
]

NUMERIC_COLS = ["age", "avg_glucose_level", "bmi", "cigsPerDay", "sleep_hours", "alcohol_drinks_per_week"]
CATEGORICAL_COLS = [
    "gender",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "residence_type",
    "dizziness",
    "chest_pain",
    "breath_shortness",
    "fatigue",
    "activity_level",
    "stress_level",
]


def _download_brfss_2022_if_needed(base_dir: str) -> str | None:
    zip_path = os.path.join(base_dir, BRFSS_2022_ZIP_NAME)
    xpt_path = os.path.join(base_dir, BRFSS_2022_XPT_NAME)

    if os.path.exists(xpt_path):
        return xpt_path

    print("BRFSS 2022 dataset not found locally. Attempting download...")
    try:
        urllib.request.urlretrieve(BRFSS_2022_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            xpt_members = [name for name in archive.namelist() if name.lower().endswith(".xpt")]
            if not xpt_members:
                print("No XPT file found inside BRFSS zip.")
                return None
            source_member = xpt_members[0]
            archive.extract(source_member, base_dir)
            extracted_path = os.path.join(base_dir, source_member)
            if extracted_path != xpt_path:
                if os.path.exists(xpt_path):
                    os.remove(xpt_path)
                os.replace(extracted_path, xpt_path)
        print(f"BRFSS 2022 downloaded: {xpt_path}")
        return xpt_path
    except Exception as exc:
        print(f"Could not download BRFSS 2022: {exc}")
        return None


def _pick_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        y_pred = (y_prob >= threshold).astype(int)
        score = 0.6 * f1_score(y_true, y_pred) + 0.4 * balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _prepare_synthetic(df_main: pd.DataFrame) -> pd.DataFrame:
    s = df_main.copy()
    status_map = {"smokes": 15, "formerly smoked": 5, "never smoked": 0, "Unknown": 0}
    s["cigsPerDay"] = s["smoking_status"].map(status_map).fillna(0)
    s["residence_type"] = s.get("Residence_type")

    for sym in ["dizziness", "chest_pain", "breath_shortness", "fatigue"]:
        s[sym] = 0
    s["sleep_hours"] = np.nan
    s["activity_level"] = np.nan
    s["alcohol_drinks_per_week"] = np.nan
    s["stress_level"] = np.nan

    s = s[[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "cigsPerDay",
        "dizziness",
        "chest_pain",
        "breath_shortness",
        "fatigue",
        "sleep_hours",
        "activity_level",
        "alcohol_drinks_per_week",
        "stress_level",
        "stroke",
    ]]
    s["source"] = "synthetic"
    return s


def _prepare_symptoms(df_symptoms: pd.DataFrame) -> pd.DataFrame:
    s = df_symptoms.copy().rename(
        columns={
            "Age": "age",
            "At Risk (Binary)": "stroke",
            "Dizziness": "dizziness",
            "Chest Pain": "chest_pain",
            "Shortness of Breath": "breath_shortness",
            "Fatigue & Weakness": "fatigue",
        }
    )

    missing_as_nan = [
        "gender",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "cigsPerDay",
        "sleep_hours",
        "activity_level",
        "alcohol_drinks_per_week",
        "stress_level",
    ]
    for col in missing_as_nan:
        if col not in s.columns:
            s[col] = np.nan

    s = s[[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "cigsPerDay",
        "dizziness",
        "chest_pain",
        "breath_shortness",
        "fatigue",
        "sleep_hours",
        "activity_level",
        "alcohol_drinks_per_week",
        "stress_level",
        "stroke",
    ]]
    s["source"] = "symptoms"
    return s


def _prepare_framingham(df_framingham: pd.DataFrame) -> pd.DataFrame:
    f = df_framingham.copy().rename(
        columns={
            "AGE": "age",
            "CIGPDAY": "cigsPerDay",
            "BMI": "bmi",
            "GLUCOSE": "avg_glucose_level",
            "HYPERTEN": "hypertension",
            "SEX": "gender",
            "STROKE": "stroke",
            "PREVCHD": "heart_disease",
        }
    )

    f["gender"] = f["gender"].map({1: "Male", 2: "Female"})
    # These fields are unavailable in Framingham; keeping NaN avoids injecting fake certainty.
    f["ever_married"] = np.nan
    f["work_type"] = np.nan
    f["residence_type"] = np.nan
    f["dizziness"] = np.nan
    f["chest_pain"] = np.nan
    f["breath_shortness"] = np.nan
    f["fatigue"] = np.nan
    f["sleep_hours"] = np.nan
    f["activity_level"] = np.nan
    f["alcohol_drinks_per_week"] = np.nan
    f["stress_level"] = np.nan

    f = f[[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "cigsPerDay",
        "dizziness",
        "chest_pain",
        "breath_shortness",
        "fatigue",
        "sleep_hours",
        "activity_level",
        "alcohol_drinks_per_week",
        "stress_level",
        "stroke",
    ]]
    f["source"] = "framingham"
    return f


def _prepare_brfss(df_brfss: pd.DataFrame) -> pd.DataFrame:
    b = df_brfss.copy()

    required_cols = [
        "CVDSTRK3",
        "_AGE80",
        "_SEX",
        "BPHIGH6",
        "CVDINFR4",
        "CVDCRHD4",
        "MARITAL",
        "EMPLOY1",
        "_BMI5",
        "_SMOKER3",
        "SLEPTIM1",
        "EXERANY2",
        "ALCDAY4",
        "AVEDRNK3",
        "SDHSTRE1",
    ]
    missing = [col for col in required_cols if col not in b.columns]
    if missing:
        print(f"BRFSS missing expected columns: {missing}")
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["stroke", "source"])

    b = b[b["CVDSTRK3"].isin([1, 2])].copy()
    b["stroke"] = (b["CVDSTRK3"] == 1).astype(int)

    b["gender"] = b["_SEX"].map({1: "Male", 2: "Female"})
    b["age"] = b["_AGE80"]
    b["hypertension"] = b["BPHIGH6"].map({1: 1, 2: 1, 3: 0, 4: 0})
    b["heart_disease"] = ((b["CVDINFR4"] == 1) | (b["CVDCRHD4"] == 1)).astype(float)
    b["ever_married"] = b["MARITAL"].map({1: "Yes", 6: "Yes", 2: "No", 3: "No", 4: "No", 5: "No"})
    b["work_type"] = b["EMPLOY1"].map({1: "Private", 2: "Private", 3: "Self-employed", 4: "Never_worked", 5: "Never_worked", 6: "Never_worked", 7: "children", 8: "Never_worked"})
    b["residence_type"] = np.nan
    b["avg_glucose_level"] = np.nan

    b["bmi"] = b["_BMI5"] / 100.0
    b.loc[(b["bmi"] <= 10) | (b["bmi"] >= 80), "bmi"] = np.nan

    b["cigsPerDay"] = b["_SMOKER3"].map({1: 15, 2: 7, 3: 2, 4: 0})
    b["dizziness"] = np.nan
    b["chest_pain"] = np.nan
    b["breath_shortness"] = np.nan
    b["fatigue"] = np.nan

    b["sleep_hours"] = b["SLEPTIM1"]
    b.loc[(b["sleep_hours"] < 1) | (b["sleep_hours"] > 24), "sleep_hours"] = np.nan

    b["activity_level"] = b["EXERANY2"].map({1: "Active", 2: "Low"})

    alc = b["ALCDAY4"]
    week_freq = np.where((alc >= 101) & (alc <= 199), alc - 100, np.nan)
    week_freq = np.where((alc >= 201) & (alc <= 299), (alc - 200) / 4.345, week_freq)
    week_freq = np.where((alc >= 301) & (alc <= 399), (alc - 300) / 52.0, week_freq)
    week_freq = np.where(alc == 888, 0, week_freq)

    drinks_per_day = b["AVEDRNK3"].copy()
    drinks_per_day = drinks_per_day.where((drinks_per_day >= 1) & (drinks_per_day <= 76), np.nan)
    b["alcohol_drinks_per_week"] = week_freq * drinks_per_day
    b["alcohol_drinks_per_week"] = b["alcohol_drinks_per_week"].clip(lower=0, upper=70)

    b["stress_level"] = b["SDHSTRE1"].map({1: "Very High", 2: "High", 3: "Medium", 4: "Low", 5: "Very Low"})

    out = b[[
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "residence_type",
        "avg_glucose_level",
        "bmi",
        "cigsPerDay",
        "dizziness",
        "chest_pain",
        "breath_shortness",
        "fatigue",
        "sleep_hours",
        "activity_level",
        "alcohol_drinks_per_week",
        "stress_level",
        "stroke",
    ]].copy()

    if len(out) > BRFSS_MAX_ROWS:
        out = out.sample(n=BRFSS_MAX_ROWS, random_state=42)

    out["source"] = "brfss2022"
    return out


def run_2_0_pipeline() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    framingham_file = os.path.join(base_dir, "Framingham Dataset.csv")
    symptoms_file = os.path.join(base_dir, "stroke_risk_dataset.csv")
    synthetic_file = os.path.join(base_dir, "synthetic_stroke_data.csv")
    brfss_xpt_file = os.path.join(base_dir, BRFSS_2022_XPT_NAME)

    print("Loading datasets...")
    df_f = pd.read_csv(framingham_file) if os.path.exists(framingham_file) else None
    df_s = pd.read_csv(symptoms_file) if os.path.exists(symptoms_file) else None
    df_main = pd.read_csv(synthetic_file) if os.path.exists(synthetic_file) else None
    df_brfss = None

    if not os.path.exists(brfss_xpt_file):
        downloaded = _download_brfss_2022_if_needed(base_dir)
        if downloaded:
            brfss_xpt_file = downloaded
    if os.path.exists(brfss_xpt_file):
        try:
            print("Loading BRFSS 2022 XPT...")
            df_brfss = pd.read_sas(brfss_xpt_file, format="xport")
        except Exception as exc:
            print(f"Could not load BRFSS 2022 XPT: {exc}")

    if df_f is None and df_s is None and df_main is None and df_brfss is None:
        print("Error: No datasets found in the project folder.")
        return

    processed = []
    if df_main is not None:
        print(f"Processing synthetic dataset ({len(df_main)} rows)...")
        processed.append(_prepare_synthetic(df_main))
    if df_s is not None:
        print(f"Processing symptom dataset ({len(df_s)} rows)...")
        processed.append(_prepare_symptoms(df_s))
    if df_f is not None:
        print(f"Processing Framingham dataset ({len(df_f)} rows)...")
        processed.append(_prepare_framingham(df_f))
    if df_brfss is not None:
        print(f"Processing BRFSS 2022 dataset ({len(df_brfss)} rows)...")
        brfss_prepared = _prepare_brfss(df_brfss)
        if not brfss_prepared.empty:
            processed.append(brfss_prepared)
            print(f"Included BRFSS training rows: {len(brfss_prepared)}")

    final_df = pd.concat(processed, ignore_index=True)
    final_df["stroke"] = final_df["stroke"].fillna(0).astype(int)
    final_df = final_df.dropna(subset=["age"])
    print(f"Unified training pool: {len(final_df)} rows")

    X = final_df[FEATURE_COLUMNS]
    y = final_df["stroke"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_COLS,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLS,
            ),
        ]
    )

    base_model = HistGradientBoostingClassifier(
        random_state=42,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=0.2,
        max_iter=450,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])

    print("Training calibrated classifier...")
    calibrated_model = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)

    valid_prob = calibrated_model.predict_proba(X_valid)[:, 1]
    threshold = _pick_threshold(y_valid, valid_prob)

    test_prob = calibrated_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    print(f"Validation-tuned threshold: {threshold:.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, test_prob):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, test_prob):.4f}")
    print(f"F1 @ threshold: {f1_score(y_test, test_pred):.4f}")
    print(f"Balanced Accuracy @ threshold: {balanced_accuracy_score(y_test, test_pred):.4f}")

    model_bundle = {
        "model": calibrated_model,
        "threshold": threshold,
        "features": FEATURE_COLUMNS,
        "model_version": "3.1-calibrated-brfss-sleep",
    }

    print("Saving model bundle...")
    joblib.dump(model_bundle, os.path.join(base_dir, "stroke_model_bundle_v3.joblib"))

    # Keep legacy artefacts for backward compatibility with older code paths.
    joblib.dump(calibrated_model, os.path.join(base_dir, "stroke_model_v2.joblib"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(base_dir, "features_v2.joblib"))

    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    run_2_0_pipeline()
