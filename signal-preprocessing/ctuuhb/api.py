import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn

from model import AFFClassifier
from config import MODEL_SAVE_PATH, PROJECT_ROOT

app = FastAPI(title="CTG Fetal Health Classifier", version="1.0")

model = None


class PredictionResult(BaseModel):
    label: int
    class_name: str
    probability_normal: float
    probability_abnormal: float


def load_model():
    global model
    model_path = MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    model = AFFClassifier()
    model.load_weights(model_path)
    model.compile()
    print(f"Model loaded from {model_path}")


@app.on_event("startup")
def startup():
    load_model()


@app.get("/")
def root():
    return {"message": "CTG Fetal Health Classifier API. POST /predict with 3 CSV files."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResult)
async def predict(
    fhr: UploadFile = File(...),
    uc: UploadFile = File(...),
    fm: UploadFile = File(...),
):
    import pandas as pd

    def _read_csv(file: UploadFile) -> np.ndarray:
        content = file.file.read()
        import io
        df = pd.read_csv(io.BytesIO(content), header=None)
        return df.values.reshape(1, 900, 1).astype(np.float32)

    fhr_arr = _read_csv(fhr)
    uc_arr = _read_csv(uc)
    fm_arr = _read_csv(fm)

    y_prob = model.predict([fhr_arr, uc_arr, fm_arr], verbose=0)[0]
    y_pred = int(np.argmax(y_prob))

    return PredictionResult(
        label=y_pred,
        class_name="正常" if y_pred == 0 else "异常",
        probability_normal=float(y_prob[0]),
        probability_abnormal=float(y_prob[1]),
    )


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
