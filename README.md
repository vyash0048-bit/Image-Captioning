# Vision Captioner

A production-oriented PyTorch image-captioning service. It uses the classic image-captioning pattern popularized by COCO projects: a pretrained ResNet-50 image encoder feeds an LSTM language decoder. The implementation is new code, rather than a clone of the reference project.

## What is included

- A testable `src/` Python package with a COCO annotation loader and deterministic vocabulary.
- Reproducible train steps via DVC, configurable artifacts, checkpoint metadata, and seed control.
- FastAPI inference with upload type/size validation, health/readiness endpoints, and container deployment.
- CI quality gates (Ruff and pytest) and pre-commit support.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,mlops]"
# Place MS-COCO images and annotations under data/raw/coco/ (ignored by Git).
captioner build-vocab --annotations data/raw/coco/annotations/captions_train2014.json
captioner train-model --images data/raw/coco/train2014 --annotations data/raw/coco/annotations/captions_train2014.json --epochs 10
uvicorn captioner.api:app --reload
```

Then send an image to `POST /v1/captions` as multipart field `image`. Swagger UI is at `http://localhost:8000/docs`.

## MLOps workflow

1. Store data outside Git and configure a DVC remote (`dvc remote add -d storage <remote>`).
2. Run `dvc repro` to make the vocabulary and model lineage reproducible.
3. Track training runs with MLflow in your deployment environment (the `mlops` extra installs the integration dependencies).
4. Promote only evaluated artifacts into the `artifacts/` deployment bundle, build the image with `docker compose up --build`, and use `/readyz` for rollout gating.

The current baseline is deliberately modest and transparent; the model boundary can later be swapped for a Transformer/BLIP-style decoder while keeping data, artifact, and service contracts stable.
