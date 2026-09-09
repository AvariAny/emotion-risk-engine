# Emotion Risk Engine

## Project Overview

Emotion Risk Engine is an AI-powered backend capable of estimating emotional risk from natural language conversations using Transformer-based models.

The system classifies text into four risk levels and is designed to be integrated into chatbots, mental health platforms, or monitoring systems.

---

## Vision

Build a modular and explainable emotional risk analysis engine using modern NLP techniques.

---

## Main Goals

- Detect emotional risk from text.
- Classify into four predefined risk levels.
- Expose predictions through a REST API.
- Keep the system modular and extensible.
- Support future explainability and model improvements.

---

## Architecture

Current architecture:

Dataset
↓

Data Cleaning

↓

DistilBERT Model

↓

Risk Classification

↓

Prediction Service

↓

REST API (Upcoming)

↓

Frontend (Upcoming)

---

## Tech Stack

- Python 3.14
- PyTorch
- Hugging Face Transformers
- Datasets
- Scikit-Learn
- Pandas
- FastAPI (planned)
- Docker (planned)

---

## Dataset

Current dataset:

- 2010 manually annotated examples
- Four balanced classes
- CSV format
- Human-reviewed annotations

Risk levels:

0 — No Risk

1 — Low Risk

2 — High Risk

3 — Critical Risk

---

## Machine Learning Model

Model:

- DistilBERT
- Sequence Classification
- Four output classes

Current performance:

Accuracy: ~82%

Weighted F1: ~82%

---

## Risk Engine

Current capabilities:

✔ Load trained model

✔ Predict emotional risk

✔ Interactive console predictions

Future capabilities:

- Confidence score
- Explainability
- Conversation context
- Rule-based post-processing

---

## API

Status:

Planned

Endpoints:

POST /predict

GET /health

GET /version

---

## Frontend

Status:

Planned

Future interface:

- Text input
- Risk visualization
- Confidence indicator

---

## Roadmap

Sprint 0 — Foundation

[x] Repository

[x] Documentation

[x] Dataset

[x] Model Training

[x] Model Evaluation

[x] Prediction Script

[ ] REST API

[ ] Frontend

[ ] Docker

[ ] Deployment

---

## Current Status

Machine Learning pipeline completed.

Current phase:

API Development.