# Heart Disease Prediction API

Assignment 17: a Random Forest classifier served using FastAPI,
containerized with Docker, and deployed on Render.

This project is an educational demonstration, not a medical tool.

## Dataset

Source:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Download and extract heart.csv into data/heart.csv.

Target mapping used:
- 0: absence of heart disease
- 1: presence of heart disease

## Input features

age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal

Categorical values use the encoding of the linked dataset.

## Model and evaluation

Model: RandomForestClassifier with 200 trees.

Exact duplicate rows are removed before a stratified 80/20
training/test split. The random seed is 42.

Measured test accuracy and classification metrics are saved in
model/metrics.json after training.

## Train locally

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
```

## Run locally without Docker

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs

## Run with Docker

Train the model before building the image.

```bash
docker compose build
docker compose up
```

Open http://localhost:8000/docs

Stop with:

```bash
docker compose down
```

## Endpoints

- GET /health: application health
- GET /info: model information and feature list
- POST /predict: returns a heart_disease boolean

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

The response contains heart_disease: true or false according
to the saved model's prediction.

## Deployment

Deploy as a Render Web Service using the Docker runtime.

Dockerfile: ./Dockerfile
Docker build context: .
Health check path: /health

Live URL: REPLACE_WITH_YOUR_RENDER_URL

Swagger UI: REPLACE_WITH_YOUR_RENDER_URL/docs