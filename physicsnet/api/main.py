from fastapi import FastAPI, HTTPException, Depends
from schemas import ParticleFeatures, PredictionResponse
from predictor import ParticlePredictor, get_predictor

app = FastAPI(
    title="PhysicsNet Inference API",
    description="Real-time particle classification using an ML ensemble",
    version="1.0.0",
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model/info")
def model_info(predictor: ParticlePredictor = Depends(get_predictor)):
    return {
        "features": predictor.features,
        "classes":  predictor.classes,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload:   ParticleFeatures,
    predictor: ParticlePredictor = Depends(get_predictor),
):
    try:
        result = predictor.predict(payload.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(
    payloads:  list[ParticleFeatures],
    predictor: ParticlePredictor = Depends(get_predictor),
):
    return [predictor.predict(p.model_dump()) for p in payloads]