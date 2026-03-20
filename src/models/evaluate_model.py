# Importamos las funciones necesarias para evaluar el modelo
# accuracy_score: calcula el porcentaje de predicciones correctas
# confusion_matrix: crea la matriz de confusión mostrando aciertos y errores
# classification_report: genera un reporte con precision, recall y F1-score
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
# Definimos la función evaluate que recibe:
# model: el modelo de regresión logística entrenado
# X_test: los datos de prueba (input)
# y_test: los resultados reales (target)
def evaluate(model, X_test, y_test):
# Usamos el modelo para predecir los resultados en X_test
# y_pred será un array con valores 0 o 1 según lo predicho
    y_pred = model.predict(X_test)
# Calculamos la exactitud del modelo
# Accuracy = (número de predicciones correctas) / (total de predicciones)
    acc = accuracy_score(y_test, y_pred)
# Generamos la matriz de confusión
# Muestra los verdaderos positivos (TP), verdaderos negativos (TN),
# falsos positivos (FP) y falsos negativos (FN)
    cm = confusion_matrix(y_test, y_pred)
# Generamos un reporte de clasificación más completo

# Incluye metrics para cada clase: precision, recall, f1-score ysupport
    report = classification_report(y_test, y_pred)
# Devolvemos las tres métricas para poder usarlas después
    return acc, cm, report