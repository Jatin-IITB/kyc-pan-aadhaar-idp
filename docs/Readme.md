# 🆔 KYC-IDP: Enterprise Grade Intelligent Document Processing

[![Status](https://img.shields.io/badge/Status-Alpha%20v2.0-orange)](https://github.com/your-username/kyc-idp)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Architecture](https://img.shields.io/badge/Architecture-Event%20Driven%20Microservices-green)](docs/architecture.md)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **Vision**: An autonomous, self-healing document extraction platform capable of processing millions of Indian KYC documents (PAN, Aadhaar) daily with >99% accuracy.

---

## 🏗️ System Architecture

We employ a decoupled **Event-Driven Microservices** architecture designed for horizontal scaling. The system separates high-throughput ingestion (API) from heavy computational inference (Workers), connected via a robust message broker (Redis).

```mermaid
graph LR
    User[Client / API] -->|1. POST /batches| Gateway[FastAPI Gateway]
    Gateway -->|2. Push Job| Broker[(Redis Broker)]
    
    subgraph "Async Inference Cluster"
        Worker[Celery Worker] -->|3. Pull Job| Broker
        Worker -->|4. Detect & Classify| YOLO[YOLOv8]
        Worker -->|5. OCR & Extract| OCR[PaddleOCR]
        Worker -->|6. Normalize| Logic[Business Rules]
    end
    
    Worker -->|7. Save Result| Storage[(S3 / FileSystem)]
    
    subgraph "Active Learning Loop (Self-Healing)"
        Worker -- "Low Confidence" --> ReviewQ[Review Queue]
        ReviewQ -->|8. Manual Fix| Analyst[Human Reviewer]
        Analyst -->|9. Gold Data| GT_DB[(Ground Truth DB)]
        GT_DB -->|10. Retrain| Trainer[Auto-FineTuner]
        Trainer -->|11. Update Weights| YOLO
    end

🧩 Core Components
Component	Tech Stack	Responsibility
Ingestion Gateway	FastAPI, Uvicorn	High-concurrency async API for job submission & status polling.
Inference Engine	Celery, PyTorch, PaddleOCR	Heavy-lifting workers that load ML models into memory once and process jobs indefinitely.
Message Broker	Redis	Decouples ingestion from processing to handle backpressure during traffic spikes.
Review Console	Streamlit	Human-in-the-Loop (HITL) interface for correcting low-confidence predictions.
Self-Healing Loop	Planned	Automated retraining pipeline that consumes corrections from the Review Console.
🚀 Quick Start Guide
Prerequisites

    Docker Desktop (for Redis infrastructure)

    Python 3.10+ (Virtual Environment recommended)

    Git

Step 1: Infrastructure Setup

Start the message broker backbone.

bash
# Start Redis container in detached mode
docker run -d -p 6379:6379 --name kyc-redis redis:7-alpine

# Verify connection
docker exec -it kyc-redis redis-cli ping  # Should return PONG

Step 2: Environment Setup

bash
# Clone the repository
git clone https://github.com/your-username/kyc-idp.git
cd kyc-idp

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Step 3: Launch Microservices (3-Terminal Setup)

You need to run three separate processes to emulate a production cluster.

Terminal 1: The Brain (Inference Worker)
Loads heavy ML models (YOLO/OCR). Wait ~20s for "Ready" logs.

bash
# Windows users need --pool=solo
python -m celery -A apps.workers.celery_main.celery_app worker --pool=solo --loglevel=INFO

Terminal 2: The Mouth (API Gateway)
Exposes REST endpoints to the outside world.

bash
python -m uvicorn apps.api_gateway.main:app --host 127.0.0.1 --port 8000 --reload

    Swagger Docs: http://localhost:8000/docs

Terminal 3: The Eyes (Review Console)
Internal dashboard for Data Analysts.

bash
streamlit run apps/review_ui/main.py

    Dashboard: http://localhost:8501

🧪 Evaluation & Benchmarking
Run Stress Test

Submit a test dataset to measure throughput and accuracy.

bash
python tools/eval_harness/run_eval.py \
  --test-dir data/test_cases_async \
  --out-dir runs/eval_latest \
  --batch-size 8 \
  --timeout-s 600

Quality Gate Check

Verify if the current build meets release criteria.

bash
python tools/eval_harness/quality_gate.py \
  --metrics runs/eval_latest/metrics.json \
  --baseline data/baselines/ci_golden_metrics.json

🔮 The Self-Healing Pipeline (Next Milestone)

We are currently building the Active Learning Loop. This will close the loop between the Review Console and the Inference Engine.

    Trigger: Docs with < 90% confidence are flagged as NEEDS_REVIEW.

    Correction: Analyst fixes the data in the Streamlit UI.

    Capture: Corrected JSON is saved to data/reviewed_ground_truth/.

    Retraining: A nightly job (tools/train/fine_tune.py) will:

        Load the new "Gold" data.

        Fine-tune the YOLOv8 classifier.

        Commit new weights to models/vNext/.

    Deploy: Workers automatically hot-reload the new model.
