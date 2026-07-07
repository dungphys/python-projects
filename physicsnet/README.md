# Project: PhysicsNet — Particle Classification & Anomaly Detection API

> A production-ready ML system that trains a classifier on physics-inspired data,
serves predictions via a REST API, tracks experiments, and runs entirely in Docker.
---

## 1. Project Overview and Architecture

**What it does**


- Generates/ingests a physics-inspired dataset (particle features)
- Trains a Random Forest + Neural Network ensemble classifier
- Tracks experiments with MLflow 
- Serves real-time predictions via a FastAPI REST endpoint
- Visualizes results via a Streamlit dashboard
- Everything runs in isolated Docker containers via Docker Compose


**Architecture**
```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                       │
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │   trainer    │────▶│   mlflow     │  :5000           │
│  │  (one-shot)  │     │  (tracking)  │                  │
│  └──────┬───────┘     └──────────────┘                  │
│         │ saves model                                   │
│         ▼                                               │
│  ┌──────────────┐     ┌──────────────┐                  │
│  │  api server  │◀────│  dashboard   │  :8501           │
│  │  (FastAPI)   │     │  (Streamlit) │                  │
│  └──────────────┘     └──────────────┘                  │
│       :8000                                             │
│                                                         │
│  ┌──────────────┐                                       │
│  │  postgres    │  (MLflow backend store)               │
│  └──────────────┘  :5432                                │
└─────────────────────────────────────────────────────────┘
```

---
 
## 2. Project Structure
 
```
physicsnet/
├── trainer/
│   ├── tuning/
│   |   ├── grid_search.py
│   |   ├── random_search.py
│   |   ├── optuna_search.py
│   |   └── validator.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train.py
│   ├── model.py
│   ├── feature_engineering.py
│   ├── test_tuning.py
│   └── data_generator.py
│
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── schemas.py
│   └── predictor.py
│
├── dashboard/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
│
├── shared/
│   └── models/              # volume-mounted: trainer writes, API reads
│
├── docker-compose.yml
├── .env
└── README.md
```
---

## 3. 