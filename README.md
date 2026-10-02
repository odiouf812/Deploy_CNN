# Plant Seedling Classifier API

Docker-ready FastAPI service for your CNN (12 weed/crop species).

## 1. Add your model
Download `model.keras` from Colab / Google Drive and put it here:

    models/model.keras

## 2. Run
    docker compose up --build

- Web app (drag & drop): http://localhost:8000
- Interactive API docs:  http://localhost:8000/docs

## 3. Use the API
    curl -X POST http://localhost:8000/predict -F "file=@seedling.png"

Response:

    {
      "filename": "seedling.png",
      "prediction": "Maize",
      "kind": "crop",
      "confidence": 0.9731,
      "top3": [{"species": "Maize", "probability": 0.9731}, ...]
    }

| Endpoint | Purpose |
|---|---|
| `GET /` | Web interface |
| `POST /predict` | One image |
| `POST /predict/batch` | Up to 20 images (`files` field) |
| `GET /classes` | The 12 species |
| `GET /health` | Status + model loaded? |

## Notes
- Preprocessing matches your notebook exactly (BGR via OpenCV, 150x150, pixel values 0-255, no rescaling). If you retrain with `/255`, change `preprocess()` in `app/main.py`.
- In your notebook the loss is `binary_crossentropy` with a 12-class softmax. Use `categorical_crossentropy` when retraining; it is the correct loss and usually improves accuracy.
- Training accuracy was only monitored; you plot no validation curve. Consider adding `ImageDataGenerator` augmentation.
