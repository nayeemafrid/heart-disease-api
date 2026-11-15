import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "model"

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found. Put heart.csv here: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = FEATURES + ["target"]
    missing = [name for name in required if name not in df.columns]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    df = df[required].apply(pd.to_numeric, errors="raise")

    if df.isna().any().any():
        raise ValueError(
            "Dataset contains missing values. Inspect and clean it first."
        )

    original_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    if set(df["target"].unique()) != {0, 1}:
        raise ValueError("Expected target labels 0 and 1.")

    X = df[FEATURES]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "model_type": "RandomForestClassifier",
        "rows_before_deduplication": original_rows,
        "rows_after_deduplication": len(df),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "test_accuracy": float(accuracy),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "positive_label": 1,
            "model_type": "RandomForestClassifier",
        },
        MODEL_DIR / "heart_model.joblib",
    )

    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Removed duplicate rows: {original_rows - len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test, predictions, zero_division=0
        )
    )
    print("Saved model/heart_model.joblib")
    print("Saved model/metrics.json")


if __name__ == "__main__":
    main()