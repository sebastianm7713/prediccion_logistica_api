import joblib
from src.data.load_data import load_data
from src.data.preprocess import prepare_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate
def run_training():
    df = load_data("data/estudiantes_examen.csv")
    X, y = prepare_data(df)
    model, X_test, y_test = train_model(X, y)
    acc, cm, report = evaluate(model, X_test, y_test)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    joblib.dump(model, "models/modelo_logistico.pkl")
    print("Modelo guardado correctamente")
if __name__ == "__main__":
    run_training()
#Un pipeline es una secuencia automatizada de procesos donde la salida de cada etapa se convierte en la entrada de la siguiente para producir un resultado final.