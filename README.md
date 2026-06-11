# Insurance Premium Prediction MLOps Project

## Overview

This repository contains an end‑to‑end machine‑learning pipeline that predicts annual health‑insurance premiums.  The project showcases a full **MLOps** workflow including:

- Data ingestion, preprocessing, and feature engineering
- Model training, evaluation, and hyper‑parameter tuning (XGBoost, LightGBM, etc.)
- Experiment tracking with **MLflow**
- Containerisation using **Docker** (API backend & Streamlit UI)
- CI/CD skeleton (GitHub Actions) – ready to be extended
- Deployment guidance for **AWS** (ECS/ECR/Fargate)

## Repository Structure
```
insurance-premium-prediction-mlops/
├─ data/                     # Sample data (git‑ignored)
├─ src/                     # Core Python modules
│   ├─ __init__.py
│   ├─ features.py           # Feature engineering utilities
│   ├─ train.py              # Training script
│   ├─ predict.py            # Inference endpoint (FastAPI)
│   └─ ...
├─ streamlit_app/            # Streamlit UI
│   └─ app.py
├─ Dockerfile.api            # API container
├─ Dockerfile.ui             # UI container
├─ .gitignore                # Ignores data, model artefacts, Insurance.txt
├─ requirements.txt          # Python dependencies
└─ README.md                 # <-- this file
```

## Quick Start (local)
```powershell
# 1. Clone the repo (if you haven't already)
git clone https://github.com/JaywardhanYadav/insurance-premium-prediction-mlops.git
cd insurance-premium-prediction-mlops

# 2. Create a virtual environment and install deps
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 3. Train the model (example)
python src/train.py

# 4. Run the API
uvicorn src.predict:app --reload --port 8501

# 5. In another terminal, launch the UI
streamlit run streamlit_app/app.py
```

## Docker (optional)
```powershell
# Build API container
docker build -f Dockerfile.api -t insurance-api .
# Build UI container
docker build -f Dockerfile.ui -t insurance-ui .
# Run containers (example)
 docker compose up -d
```


