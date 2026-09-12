# Emotion Risk Engine Roadmap

## Vision

Emotion Risk Engine aims to become a complete AI-powered emotional risk assessment platform.

The project will evolve from a machine learning model into a deployable product composed of backend, frontend, mobile application, and cloud infrastructure.

---

# Phase 1 — Machine Learning ✅

Status: Completed

Objectives:

- Build dataset
- Train DistilBERT
- Evaluate model
- Interactive prediction
- Documentation
- GitHub repository

---

# Phase 2 — Backend API

Objectives:

- FastAPI
- REST API
- Swagger documentation
- Confidence scores
- Health endpoint
- Version endpoint

Deliverables:

POST /predict

GET /health

GET /version

---

# Phase 3 — Database

Objectives:

Store prediction history.

Potential database:

- PostgreSQL

Stored information:

- Text (with user consent)
- Prediction
- Confidence
- Timestamp
- User feedback

---

# Phase 4 — Web Application

Technology:

- React

Goals:

- Public demo
- Text input
- Risk visualization
- Confidence display

Deployment:

- Vercel

---

# Phase 5 — Cloud Deployment

Backend:

- Render

Frontend:

- Vercel

Database:

- Supabase PostgreSQL

Goal:

Allow anyone to test the Emotion Risk Engine online.

---

# Phase 6 — Android Application

Technology:

React Native

Goals:

- Consume the same FastAPI backend.
- Mobile interface.
- Android deployment.

---

# Phase 7 — Future Improvements

Possible features:

- Explainable AI (XAI)
- Conversation context analysis
- Risk trends
- User authentication
- Dashboard
- Analytics
- Multi-language support
- Fine-tuning with real data
- Docker
- CI/CD
- Monitoring

---

# Long-Term Vision

Emotion Risk Engine should become a modular AI platform capable of being integrated into chatbots, web applications, mobile apps, and research systems.

The API will be the core component consumed by different clients:

React Web

↓

FastAPI

↓

Emotion Risk Engine

↓

DistilBERT

↓

PostgreSQL

↓

React Native