# Emotion Risk Engine

## Project Overview

Emotion Risk Engine is an AI-powered Natural Language Processing (NLP) backend capable of estimating emotional risk from Spanish-language conversations using Transformer-based models.

The engine classifies messages into four emotional risk levels and exposes predictions through a REST API, making it suitable for integration into chatbots, healthcare platforms, educational systems, or real-time monitoring services.

---

# Vision

Build a modular, explainable, and production-ready emotional risk analysis engine specialized in the Spanish language.

---

# Main Goals

- Detect emotional risk from free-text conversations.
- Classify messages into four predefined risk levels.
- Return confidence scores for every prediction.
- Expose predictions through a REST API.
- Keep the architecture modular and scalable.
- Support future explainability and conversation-level reasoning.

---

# Current Architecture

```
Dataset Generation
        │
        ▼
Dataset Cleaning
        │
        ▼
Train / Validation / Test Split
        │
        ▼
BETO (Spanish BERT)
        │
        ▼
Emotion Risk Classification
        │
        ▼
Prediction Engine
        │
        ▼
Probability Estimation
        │
        ▼
REST API (Upcoming)
        │
        ▼
Frontend / Mobile App (Upcoming)
```

---

# Tech Stack

- Python 3.14
- PyTorch
- Hugging Face Transformers
- BETO (dccuchile/bert-base-spanish-wwm-cased)
- Datasets
- Scikit-Learn
- Pandas
- NumPy
- FastAPI (planned)
- Docker (planned)

---

# Dataset

Current dataset:

- **7,819** manually reviewed Spanish chat messages
- Four balanced emotional risk classes
- CSV format
- Human-verified annotations
- Multiple everyday conversation topics

Risk levels:

| Label | Description |
|-------|-------------|
| 0 | No Risk |
| 1 | Low Risk |
| 2 | Moderate Risk |
| 3 | High Risk |

---

# Machine Learning Model

Current model

- BETO (Spanish BERT)
- Sequence Classification
- Four output classes
- Hugging Face Transformers

Current performance (Test Set)

| Metric | Score |
|--------|--------|
| Accuracy | **97.57%** |
| Precision | **97.66%** |
| Recall | **97.60%** |
| Macro F1 | **97.60%** |

---

# Current Features

✔ Dataset generation

✔ Dataset statistics

✔ Train / Validation / Test split

✔ BETO training pipeline

✔ Model evaluation

✔ Confusion matrix generation

✔ Classification report

✔ Interactive prediction

✔ Prediction with probabilities

✔ Best checkpoint saving

---

# Upcoming Features

- Probability calibration
- Explainability (Attention / SHAP)
- Conversation context
- Rule-based post-processing
- Model versioning
- Docker support

---

# REST API (Next Sprint)

Status

Planned

Endpoints

```
GET  /health
GET  /version
POST /predict
POST /predict_proba
```

---

# Frontend

Status

Planned

Future interface

- Text input
- Risk visualization
- Confidence bars
- Probability distribution
- Conversation history

---

# Mobile App

Status

Planned

Features

- Chat interface
- Real-time predictions
- REST API integration
- Emotional risk visualization

---

# Roadmap

## Sprint 0 — Dataset

- [x] Repository
- [x] Documentation
- [x] Dataset generation
- [x] Dataset statistics
- [x] Train / Validation / Test split

---

## Sprint 1 — Machine Learning

- [x] BETO training
- [x] Model evaluation
- [x] Confusion matrix
- [x] Prediction script
- [x] Prediction with probabilities

---

## Sprint 2 — Model Export

- [ ] Export trained model
- [ ] ONNX export
- [ ] TorchScript export

---

## Sprint 3 — Backend

- [ ] FastAPI
- [ ] CRUD
- [ ] Prediction endpoints
- [ ] Swagger documentation

---

## Sprint 4 — Frontend

- [ ] Web Interface
- [ ] Mobile App
- [ ] Charts
- [ ] Authentication

---

## Sprint 5 — Deployment

- [ ] Docker
- [ ] Docker Compose
- [ ] CI/CD
- [ ] Cloud Deployment

---

# Current Status

Current version:

**Emotion Risk Engine v2 (BETO)**

Pipeline status

✅ Dataset completed

✅ Model trained

✅ Model evaluated

✅ Prediction engine completed

🚧 Exporting model (next)

🚧 REST API (next)

🚧 CRUD

🚧 Frontend

🚧 Deployment