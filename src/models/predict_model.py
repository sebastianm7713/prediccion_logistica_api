import joblib
import numpy as np
model = joblib.load("models/modelo_logistico.pkl")
def predict(horas_estudio):
    data = np.array([[horas_estudio]])
    prob = model.predict_proba(data)[0][1] # probabilidad de aprobar
    pred = model.predict(data)[0] # 0 o 1
    return int(pred), float(prob)