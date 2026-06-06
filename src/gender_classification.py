from __future__ import annotations

import os
import numpy as np
import joblib
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def build_gender_classifier_pipeline(n_components: int, random_state: int) -> Any:
    """Construye el pipeline para la clasificación de género."""
    pipeline = Pipeline([
        ("pca", PCA(n_components=n_components, whiten=True, random_state=random_state)),
        ("clf", GaussianNB()),
    ])
    return pipeline

def train_gender_classifier(
    X_train: Any,
    y_gender_train: Any,
    n_components: int,
    random_state: int,
) -> Any:
    """Entrena el pipeline clasificador de género."""
    pipeline = build_gender_classifier_pipeline(n_components, random_state)
    pipeline.fit(X_train, y_gender_train)
    
    return pipeline

def evaluate_gender_classifier(model: Any, X_test: Any, y_gender_test: Any) -> dict[str, Any]:
    """Evalúa el modelo de género y retorna métricas numéricas."""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_gender_test, y_pred)
    # Generamos un reporte numérico en formato diccionario
    report = classification_report(y_gender_test, y_pred, output_dict=True)
    
    return {
        "accuracy": acc,
        "report": report
    }

def save_gender_classifier(model: Any, output_path: str) -> None:
    """Guarda el pipeline completo de género en disco."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)