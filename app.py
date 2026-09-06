from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, redirect, render_template, request, url_for

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

DISEASES = [
    ("01_heart_disease", "Heart Disease", "cardiovascular", "A focused check of cardiovascular risk signals."),
    ("02_diabetes", "Diabetes", "metabolic", "Review glucose, BMI, and lifestyle indicators."),
    ("03_breast_cancer", "Breast Cancer", "oncology", "Assess the numerical markers used by the trained model."),
    ("04_liver_disease", "Liver Disease", "organ health", "Explore the liver health indicators in your profile."),
    ("05_kidney_disease", "Kidney Disease", "organ health", "Check the lab signals used for kidney risk."),
    ("06_parkinsons", "Parkinson's Disease", "neurology", "Review the movement and neurological measurements."),
    ("07_stroke", "Stroke", "neurology", "Assess the vascular risk factors provided to the model."),
    ("08_lung_cancer", "Lung Cancer", "oncology", "Explore respiratory and lifestyle risk signals."),
    ("09_thyroid", "Thyroid Disease", "endocrine", "Review thyroid-related measurements and symptoms."),
    ("10_skin_disease", "Skin Disease", "dermatology", "Assess the lesion characteristics captured by the model."),
    ("11_alzheimers", "Alzheimer's Disease", "neurology", "Review the cognitive and brain health measurements."),
    ("12_hepatitis", "Hepatitis", "organ health", "Check the blood markers used in the prediction model."),
    ("13_malaria", "Malaria", "infectious", "Review fever, blood, and symptom indicators."),
    ("14_pneumonia", "Pneumonia", "respiratory", "Assess respiratory measurements and reported symptoms."),
    ("15_covid19", "COVID-19", "infectious", "Review the respiratory and symptom profile."),
    ("16_anemia", "Anemia", "blood health", "Check the blood measurements used by the model."),
    ("17_obesity", "Obesity", "metabolic", "Explore body composition and activity indicators."),
    ("18_asthma", "Asthma", "respiratory", "Review breathing symptoms and peak-flow data."),
    ("19_dengue", "Dengue", "infectious", "Assess symptom and blood-count indicators."),
    ("20_pcos", "PCOS", "hormonal health", "Review cycle, hormone, and metabolic measurements."),
]


def pretty_label(name):
    return name.replace("_", " ").title()


def load_disease(slug):
    artifact = joblib.load(BASE_DIR / f"{slug}_random_forest.pkl")
    dataset = pd.read_csv(BASE_DIR / f"{slug}.csv")
    features = []

    for name in artifact["feature_columns"]:
        series = dataset[name]
        numeric = pd.to_numeric(series, errors="coerce")
        unique_values = sorted(numeric.dropna().unique().tolist())
        is_binary = set(unique_values).issubset({0, 1}) and len(unique_values) <= 2
        median = float(numeric.median()) if not numeric.dropna().empty else 0.0
        features.append({
            "name": name,
            "label": pretty_label(name),
            "is_binary": is_binary,
            "example": int(round(median)) if is_binary else round(median, 3),
            "min": round(float(numeric.min()), 3) if not numeric.dropna().empty else 0,
            "max": round(float(numeric.max()), 3) if not numeric.dropna().empty else 100,
            "step": 1 if is_binary or pd.api.types.is_integer_dtype(series) else "any",
        })

    return artifact, features


def disease_card(slug, name, category, description):
    return {"slug": slug, "name": name, "category": category, "description": description}


DISEASE_CARDS = [disease_card(*disease) for disease in DISEASES]
DISEASE_LOOKUP = {disease[0]: disease_card(*disease) for disease in DISEASES}


@app.get("/")
def home():
    return render_template("index.html", diseases=DISEASE_CARDS)


@app.route("/predict/<slug>", methods=["GET", "POST"])
def predict(slug):
    disease = DISEASE_LOOKUP.get(slug)
    if disease is None:
        abort(404)

    try:
        artifact, features = load_disease(slug)
    except FileNotFoundError:
        abort(500, description=f"Model files for {slug} are missing.")

    prediction = None
    probability = None
    error = None
    submitted_values = {}

    if request.method == "POST":
        try:
            submitted_values = {feature["name"]: request.form.get(feature["name"], "") for feature in features}
            values = [float(submitted_values[feature["name"]]) for feature in features]
            row = pd.DataFrame([values], columns=artifact["feature_columns"])
            prediction = int(artifact["model"].predict(row)[0])

            if hasattr(artifact["model"], "predict_proba"):
                probabilities = artifact["model"].predict_proba(row)[0]
                positive_index = list(artifact["model"].classes_).index(1)
                probability = round(float(probabilities[positive_index]) * 100, 1)
        except (TypeError, ValueError, KeyError):
            error = "Please enter a valid value in every field."

    return render_template(
        "predict.html",
        disease=disease,
        features=features,
        prediction=prediction,
        probability=probability,
        error=error,
        submitted_values=submitted_values,
    )


@app.get("/health")
def health():
    return {"status": "ok", "models": len(DISEASE_CARDS)}


if __name__ == "__main__":
    app.run(debug=True)
