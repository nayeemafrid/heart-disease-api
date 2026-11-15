from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from app.schemas import HeartInput, PredictionResponse


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "heart_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file missing. Run python train_model.py first."
        )

    app.state.bundle = joblib.load(MODEL_PATH)
    yield
    del app.state.bundle


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Assignment 17: FastAPI, Docker, and cloud deployment. "
        "Educational demonstration only; not medical advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/info")
def info(request: Request):
    bundle = request.app.state.bundle

    return {
        "model_type": bundle["model_type"],
        "features": bundle["features"],
        "target_mapping": {
            "0": "absence of heart disease",
            "1": "presence of heart disease",
        },
        "note": "Educational demonstration only.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HeartInput, request: Request):
    bundle = request.app.state.bundle

    input_data = pd.DataFrame(
        [payload.model_dump()],
        columns=bundle["features"],
    )

    prediction = bundle["model"].predict(input_data)[0]

    return PredictionResponse(
        heart_disease=bool(
            prediction == bundle["positive_label"]
        )
    )