from __future__ import annotations

import io
import json
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartcrop")

app = FastAPI(title="Smart Crop Health AI", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class CropInput(BaseModel):
    N: float = Field(ge=0, le=200)
    P: float = Field(ge=0, le=200)
    K: float = Field(ge=0, le=250)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=2000)
    target_crop: str | None = Field(default=None, max_length=80)


@lru_cache(maxsize=1)
def load_crop_model():
    path = MODEL_DIR / "crop_risk.joblib"
    if not path.is_file():
        raise FileNotFoundError("Crop ML checkpoint is missing")
    return joblib.load(path)


@lru_cache(maxsize=1)
def load_leaf_model():
    import torch
    from src.leaf_model import LeafCNN

    checkpoint = MODEL_DIR / "leaf_cnn.pt"
    classes_path = MODEL_DIR / "leaf_classes.json"
    if not checkpoint.is_file() or not classes_path.is_file():
        raise FileNotFoundError("Leaf CNN checkpoint or class map is missing")
    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("Leaf class map is invalid")
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    model = LeafCNN(len(classes))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model, classes


def probability_band(probability: float) -> str:
    if probability >= 0.60:
        return "Lower model-indicated risk"
    if probability >= 0.30:
        return "Moderate model-indicated risk"
    return "Higher model-indicated risk"


@app.get("/", response_class=FileResponse)
def home():
    return FileResponse(ROOT / "templates" / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/crops")
def crop_classes():
    try:
        model = load_crop_model()
        return {"crops": sorted(str(name) for name in model.classes_)}
    except Exception as exc:
        logger.exception("Crop model is unavailable")
        raise HTTPException(status_code=503, detail="Crop model is unavailable. Please try again later.") from exc


@app.post("/api/crop-predict")
def predict_crop(request: CropInput):
    try:
        model = load_crop_model()
        if request.target_crop:
            known_crops = {str(label).casefold() for label in model.classes_}
            if request.target_crop.strip().casefold() not in known_crops:
                raise HTTPException(status_code=422, detail="Choose a crop from the available model labels.")
        row = pd.DataFrame([{name: getattr(request, name) for name in FEATURES}], columns=FEATURES)
        probabilities = model.predict_proba(row)[0]
        ranking = sorted(
            ((str(label), float(prob)) for label, prob in zip(model.classes_, probabilities)),
            key=lambda item: item[1], reverse=True,
        )
        matches = {name.casefold(): probability for name, probability in ranking}
        selected_probability = matches.get(request.target_crop.strip().casefold()) if request.target_crop else None
        return {
            "top_recommendation": ranking[0][0],
            "recommendations": [{"crop": crop, "probability": round(prob, 5)} for crop, prob in ranking[:5]],
            "selected_crop": request.target_crop,
            "selected_crop_probability": round(selected_probability, 5) if selected_probability is not None else None,
            "risk_band": probability_band(selected_probability) if selected_probability is not None else None,
            "model_note": "Model probabilities reflect this training dataset; they are not calibrated crop-failure probabilities.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Crop prediction failed")
        raise HTTPException(status_code=503, detail="Crop prediction failed because the model is unavailable or incompatible.") from exc


@app.post("/api/leaf-predict")
async def predict_leaf(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    if Path(filename).suffix not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, JPEG, or PNG image.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The photo must be 8 MB or smaller.")
    try:
        with Image.open(io.BytesIO(content)) as check:
            check.verify()
        with Image.open(io.BytesIO(content)) as source:
            if source.width < 64 or source.height < 64:
                raise HTTPException(status_code=422, detail="Use a clear leaf photo at least 64 × 64 pixels.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="The image dimensions are too large. Choose a smaller photo.")
            image = source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="That file could not be read as a valid image. Try another JPG or PNG.") from exc

    try:
        import torch

        model, classes = load_leaf_model()
        array = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = torch.from_numpy(((array - mean) / std).transpose(2, 0, 1).copy()).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        values, indices = torch.topk(probabilities, min(3, len(classes)))
        confidence = float(values[0])
        return {
            "prediction": classes[int(indices[0])],
            "confidence": round(confidence, 5),
            "low_confidence": confidence < 0.55,
            "top_predictions": [
                {"class_name": classes[int(index)], "probability": round(float(value), 5)}
                for value, index in zip(values, indices)
            ],
            "model_note": "This classifier was trained on curated PlantVillage images and can be less accurate on field photos.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Leaf disease inference failed")
        raise HTTPException(status_code=503, detail="Leaf disease model is unavailable or could not process this image.") from exc
