# Deployment Guide for AWS

## Overview
This project consists of three main components:
1. **FastAPI backend** (`src/inference.py`, `src/train.py`, `src/app.py` via Dockerfile.api) – serves the `/predict` endpoint.
2. **Streamlit UI** (`streamlit_app/app.py`) – provides a web UI that calls the FastAPI service.
3. **MLflow tracking** – logs experiments and stores the trained model artifacts.

## Important files for deployment
| Component | File / Directory | Purpose |
|----------|-------------------|---------|
| FastAPI  | `Dockerfile.api`   | Builds the backend container with all Python dependencies. |
|          | `src/` (all `.py` files) | Source code for the API, training, inference, and utilities. |
| Streamlit UI | `Dockerfile.ui`   | Builds a lightweight container to serve the Streamlit app. |
|          | `streamlit_app/` (especially `app.py` and `assets/`) | UI code and static assets. |
| MLflow   | `mlflow/` (optional) or `mlruns/` directory | Stores experiment runs and model artifacts. |
| Config   | `config/` (e.g., `config.yaml`) | Central configuration for paths, DB connection, model hyper‑parameters, etc. |
| Docker Compose | `docker‑compose.yml` | Orchestrates the three containers (API, UI, MLflow) and sets networking. |

## AWS deployment steps (ECS/Fargate recommended)
1. **Create an ECR repository** for each container (`api`, `ui`).
   ```bash
   aws ecr create-repository --repository-name insurance-api
   aws ecr create-repository --repository-name insurance-ui
   ```
2. **Build and push images** from your local machine (or CI/CD pipeline).
   ```bash
   # Build API image
   docker build -f Dockerfile.api -t insurance-api:latest .
   # Tag & push
   $(aws ecr get-login --no-include-email)
   docker tag insurance-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/insurance-api:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/insurance-api:latest

   # Build UI image
   docker build -f Dockerfile.ui -t insurance-ui:latest .
   docker tag insurance-ui:latest <account-id>.dkr.ecr.<region>.amazonaws.com/insurance-ui:latest
   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/insurance-ui:latest
   ```
3. **Create an ECS cluster** (Fargate launch type).
   ```bash
   aws ecs create-cluster --cluster-name insurance‑cluster
   ```
4. **Define task definitions** (JSON) for the two containers.
   - **API task** – expose port `8001`, set env vars `API_URL`, `MLFLOW_TRACKING_URI`, etc.
   - **UI task** – expose port `8501`, set `API_URL` env var pointing to the API service (`http://<api‑task‑name>:8001`).
5. **Create a service** for each task definition (desired count = 1). Ensure they share a common security group that allows inbound traffic on ports `8001` and `8501`.
6. **Optional: ALB (Application Load Balancer)**
   - Add listeners on `80/443` that forward to the UI task (port `8501`).
   - The UI will call the API via the internal service name, no public exposure needed.
7. **Set up Secrets** (e.g., database credentials) in AWS Secrets Manager and reference them in the task definition.
8. **Configure CloudWatch logging** for each container to capture stdout/stderr (including training logs if you run training as a job). 
9. **Run training**
   - You can launch a one‑off ECS task that executes `python -m src.train` using the same image as the API (or a dedicated training image). This will write the model to the shared EFS volume mounted at `/app/mlruns`.
   - Alternatively, run training locally and push the produced model artifact to the S3 bucket that the API loads from.
10. **Verify**
    - Open the ALB DNS name in a browser → Streamlit UI should load.
    - Submit a sample record → UI calls FastAPI → you should see the prediction JSON.
    - Check the API logs in CloudWatch for any errors.

## Quick‑start script (optional)
A helper Bash script is provided in the repo (`scripts/deploy_aws.sh`). It automates steps 1‑6 and prints the ALB URL.
```bash
#!/usr/bin/env bash
set -e
# ... script body ...
```
Make it executable (`chmod +x scripts/deploy_aws.sh`).

## Summary of required files
```
Insurance Project/
├─ Dockerfile.api          # FastAPI container
├─ Dockerfile.ui           # Streamlit container
├─ docker-compose.yml      # Local dev orchestration (can be reused for ECS task defs)
├─ config/                # YAML config used by both services
├─ src/                   # All Python modules (API, training, utils)
├─ streamlit_app/         # Streamlit UI code and assets
├─ mlruns/                # MLflow run data (persisted via EFS/S3)
└─ scripts/deploy_aws.sh  # Convenience deployment script
```

Follow the steps above, replace placeholder `<account-id>` and `<region>` with your AWS values, and you’ll have the project running on AWS.

---
*If you need the Dockerfiles themselves or the `docker‑compose.yml` contents, let me know and I’ll paste them.*
