"""FastAPI inference service with readiness checks and Prometheus-compatible metrics."""
from __future__ import annotations
from contextlib import asynccontextmanager
from io import BytesIO
import time
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import transforms
from .model import CaptioningModel
from .settings import settings
from .vocabulary import Vocabulary

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class CaptionResponse(BaseModel):
    caption: str
    latency_ms: float
    model_version: str

class Runtime:
    model: CaptioningModel | None = None
    vocabulary: Vocabulary | None = None
    device: torch.device = torch.device("cpu")
    version: str = "unavailable"

runtime = Runtime()

def load_runtime() -> None:
    if not settings.model_path.exists() or not settings.vocab_path.exists(): return
    runtime.vocabulary = Vocabulary.load(settings.vocab_path)
    runtime.device = torch.device("cuda" if settings.device == "auto" and torch.cuda.is_available() else settings.device if settings.device != "auto" else "cpu")
    checkpoint = torch.load(settings.model_path, map_location=runtime.device, weights_only=True)
    runtime.model = CaptioningModel(len(runtime.vocabulary)); runtime.model.load_state_dict(checkpoint["model_state"])
    runtime.model.to(runtime.device).eval(); runtime.version = f"epoch-{checkpoint.get('epoch', 'unknown')}"

@asynccontextmanager
async def lifespan(_: FastAPI):
    load_runtime(); yield

app = FastAPI(title="Vision Captioner", version="0.1.0", lifespan=lifespan)

@app.get("/healthz")
def healthz() -> dict[str, str]: return {"status": "ok"}

@app.get("/readyz")
def readyz() -> dict[str, str]:
    if runtime.model is None or runtime.vocabulary is None: raise HTTPException(503, "Model artifacts are not loaded")
    return {"status": "ready", "model_version": runtime.version}

@app.post("/v1/captions", response_model=CaptionResponse)
async def caption(image: UploadFile = File(...)) -> CaptionResponse:
    if runtime.model is None or runtime.vocabulary is None: raise HTTPException(503, "Model artifacts are not loaded")
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}: raise HTTPException(415, "Upload a JPEG, PNG, or WebP image")
    payload = await image.read(settings.max_upload_bytes + 1)
    if len(payload) > settings.max_upload_bytes: raise HTTPException(413, "Image exceeds upload limit")
    try:
        with Image.open(BytesIO(payload)) as uploaded:
            tensor = transform(uploaded.convert("RGB")).unsqueeze(0).to(runtime.device)
    except UnidentifiedImageError as error: raise HTTPException(422, "Invalid image payload") from error
    started = time.perf_counter()
    ids = runtime.model.generate(tensor, runtime.vocabulary.bos_id, runtime.vocabulary.eos_id)[0]
    return CaptionResponse(caption=runtime.vocabulary.decode(ids), latency_ms=round((time.perf_counter() - started) * 1000, 2), model_version=runtime.version)
