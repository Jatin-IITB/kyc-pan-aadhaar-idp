# KYC-IDP: Intelligent Document Processing for Indian KYC

An end-to-end document extraction platform for Indian identity documents (PAN & Aadhaar) using YOLOv8 field detection, PaddleOCR, LLM-based post-processing, and a human-in-the-loop review system.

## Architecture

```
Client --> FastAPI Gateway --> Redis Broker --> Celery Worker
                                                  |
                                        +---------+---------+
                                        |         |         |
                                     YOLOv8  PaddleOCR  Validator
                                        |         |         |
                                        +----+----+----+----+
                                             |         |
                                        LLM Rescue  Schema Check
                                             |
                                     Result Storage
                                             |
                                    Streamlit Review UI
```

**Pipeline flow:** Image upload -> Quality gate (blur/exposure) -> Document classification (PAN vs Aadhaar) -> Field detection (YOLOv8 bounding boxes) -> OCR extraction (PaddleOCR) -> Normalization & validation (JSON Schema + Verhoeff checksum) -> LLM rescue for failed extractions (Ollama) -> Human review for low-confidence results.

## Tech Stack

| Layer | Technology |
|-------|------------|
| API Gateway | FastAPI + Uvicorn |
| Task Queue | Celery + Redis |
| Field Detection | YOLOv8 (custom-trained) |
| OCR Engine | PaddleOCR |
| Post-processing | Ollama (Llama 3.2) |
| Validation | JSON Schema + Verhoeff |
| Review UI | Streamlit |
| Containerization | Docker Compose |

## Quick Start

### Option A: Docker Compose (recommended)

```bash
docker-compose up
```

API at `http://localhost:8000/docs`.

### Option B: Local Development

**Prerequisites:** Python 3.10+, Redis running on port 6379.

```bash
# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Terminal 1: Worker (loads ML models, wait ~20s)
celery -A apps.workers.celery_main.celery_app worker --pool=solo --loglevel=INFO

# Terminal 2: API
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 3: Review UI
streamlit run apps/review_ui/main.py
```

## API

```bash
# Submit single document
curl -X POST http://localhost:8000/jobs \
  -F "file=@pan_card.jpg" \
  -F "doc_type=auto"

# Poll result
curl http://localhost:8000/jobs/{job_id}

# Submit batch
curl -X POST http://localhost:8000/batches \
  -F "files=@doc1.jpg" \
  -F "files=@doc2.jpg"

# Poll batch
curl http://localhost:8000/batches/{batch_id}
```

## Project Structure

```
apps/
  api/              FastAPI gateway (job submission, status polling)
  workers/          Celery tasks (ML inference pipeline)
  review_ui/        Streamlit human-in-the-loop console
  common/           Shared settings loader
services/
  pipeline.py       Main KYCPipeline orchestrator
  card_crop_yolov8/ YOLOv8 field detection
  ocr_paddle/       PaddleOCR text extraction
  doc_classifier/   PAN vs Aadhaar routing
  extraction/       Normalization + LLM cleanup
  validation/       JSON Schema enforcement
  preprocessing/    Image quality gate
  ingestion/        File storage abstraction
config/
  app.yaml          Application config
  models.yaml       ML model paths & parameters
  thresholds.yaml   Quality gate thresholds
  schemas/          JSON validation schemas (PAN, Aadhaar)
tools/
  eval_harness/     Evaluation runner + CI quality gate
  synthetic_id_generator/  Training data generation
  train/            Model fine-tuning utilities
tests/
  unit/             Pipeline unit tests
  integration/      API contract tests
```

## Evaluation

```bash
# Run evaluation suite
python tools/eval_harness/run_eval.py \
  --test-dir data/test_cases/golden \
  --out-dir runs/eval_latest \
  --batch-size 8

# Check quality gate
python tools/eval_harness/quality_gate.py \
  --metrics runs/eval_latest/metrics.json \
  --baseline data/baselines/ci_golden_metrics.json
```

## Environment Variables

See [`.env.example`](.env.example) for all configurable variables.

## License

MIT
