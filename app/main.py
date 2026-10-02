"""Plant Seedling Classifier API (FastAPI + Keras CNN)."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

IMAGE_SIZE = 150  # same as in training
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.keras")
MAX_BYTES = 10 * 1024 * 1024

# Alphabetical order = sklearn LabelEncoder order used in training.
CLASSES = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed",
    "Common wheat", "Fat Hen", "Loose Silky-bent", "Maize",
    "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
    "Sugar beet",
]
CROPS = {"Maize", "Common wheat", "Sugar beet"}

STATIC_DIR = Path(__file__).parent / "static"
state = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import keras  # imported lazily so the API starts fast
    if not Path(MODEL_PATH).exists():
        print(f"[warn] Model not found at {MODEL_PATH}. /predict will return 503.")
    else:
        state["model"] = keras.saving.load_model(MODEL_PATH, compile=False)
        print(f"[ok] Model loaded from {MODEL_PATH}")
    yield


app = FastAPI(
    title="Plant Seedling Classifier",
    description="Upload a photo of a young plant and find out whether it is a "
                "crop or a weed, and which species it is.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def preprocess(raw: bytes) -> np.ndarray:
    """Identical to training: cv2 BGR, resize 150x150, float32, no scaling."""
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "That file is not a readable image. Use PNG or JPG.")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img.astype("float32")[None, ...]


def classify(raw: bytes) -> dict:
    if state["model"] is None:
        raise HTTPException(503, "Model not loaded. Put model.keras in the models/ folder.")
    probs = state["model"].predict(preprocess(raw), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    top = CLASSES[order[0]]
    return {
        "prediction": top,
        "kind": "crop" if top in CROPS else "weed",
        "confidence": round(float(probs[order[0]]), 4),
        "top3": [
            {"species": CLASSES[i], "probability": round(float(probs[i]), 4)}
            for i in order[:3]
        ],
    }


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Status"])
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None}


@app.get("/classes", tags=["Status"])
def classes():
    return [{"species": c, "kind": "crop" if c in CROPS else "weed"} for c in CLASSES]


@app.post("/predict", tags=["Prediction"], summary="Identify a seedling from one photo")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    return {"filename": file.filename, **classify(raw)}


@app.post("/predict/batch", tags=["Prediction"], summary="Identify several photos at once")
async def predict_batch(files: list[UploadFile] = File(...)):
    if len(files) > 20:
        raise HTTPException(400, "Send at most 20 images per request.")
    out = []
    for f in files:
        raw = await f.read()
        try:
            out.append({"filename": f.filename, **classify(raw)})
        except HTTPException as e:
            out.append({"filename": f.filename, "error": e.detail})
    return out
