from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict_model import predict
app = FastAPI(
    title="Examen Prediction API",
    description="API para predecir si un estudiante aprobará según horas de estudio",
    version="1.0"
)
class EstudianteInput(BaseModel):
    horas_estudio: float
@app.get("/")

def root():
    return {"message": "API de predicción activa"}
@app.post("/predict")
def predict_exam(data: EstudianteInput):
    resultado, probabilidad = predict(data.horas_estudio)
    return {
        "horas_estudio": data.horas_estudio,
        "aprobara": bool(resultado),
        "probabilidad": probabilidad
}