import joblib
import numpy as np
from typing import Any
from src.preprocessing import preprocess_face_array

def load_models(gender_path: str, age_path: str) -> tuple[Any, Any]:
    """Carga los modelos entrenados desde el disco."""
    try:
        gender_model = joblib.load(gender_path)
        age_model = joblib.load(age_path)
        return gender_model, age_model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Falta un modelo. ¿Ejecutaste main.py primero? Error: {e}")

def predict_face(face_roi: np.ndarray, gender_model: Any, age_model: Any) -> tuple[str, int]:
    """Toma un recorte de rostro, lo preprocesa y predice género y edad."""
    # 1. Aplicamos EXACTAMENTE el mismo preprocesamiento del entrenamiento
    face_vector, _ = preprocess_face_array(face_roi)
    
    # 2. El modelo espera una matriz 2D (1 fila, 625 columnas)
    face_vector = face_vector.reshape(1, -1)
    
    # 3. Inferencia
    pred_gender = gender_model.predict(face_vector)[0]
    pred_age = age_model.predict(face_vector)[0]
    
    # 4. Mapeo (0: Mujer, 1: Hombre según vimos en tu pandas DataFrame)
    gender_label = "Mujer" if pred_gender == 0 else "Hombre"
    
    return gender_label, int(pred_age)