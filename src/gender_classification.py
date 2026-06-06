from __future__ import annotations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os
import numpy as np
import joblib
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def build_gender_classifier_pipeline(random_state: int) -> Any:
    """Construye el pipeline base para la clasificación de género."""
    pipeline = Pipeline([
        ("pca", PCA(whiten=True, random_state=random_state)),
        ("clf", GaussianNB()),
    ])
    return pipeline


def train_gender_classifier(
    X_train: Any,
    y_gender_train: Any,
    pca_components: tuple[int, ...],
    random_state: int,
) -> Any:
    """Entrena el pipeline clasificador de género evaluando múltiples hiperparámetros."""
    pipeline = build_gender_classifier_pipeline(random_state)
    
    # Definimos la grilla de parámetros a probar
    param_grid = {
        "pca__n_components": pca_components,
    }
    
    # Configuramos la competencia de modelos
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy", # Le decimos que el modelo ganador será el de mayor exactitud
        cv=5,               
        n_jobs=-1,          
    )
    
    grid_search.fit(X_train, y_gender_train)
    
    print(f"      Mejor número de componentes PCA para Género: {grid_search.best_params_['pca__n_components']}") 
    return grid_search.best_estimator_

def evaluate_gender_classifier(model: Any, X_test: Any, y_gender_test: Any) -> dict[str, Any]:
    """Evalúa el modelo de género y retorna métricas numéricas."""
    y_pred = model.predict(X_test)
    
    # Cálculos de métricas
    acc = accuracy_score(y_gender_test, y_pred)
    pre = precision_score(y_gender_test, y_pred, average="binary")
    rec = recall_score(y_gender_test, y_pred, average="binary")
    f1 = f1_score(y_gender_test, y_pred, average="binary")
    
    # Matriz de confusión
    cm = confusion_matrix(y_gender_test, y_pred)
    
    return {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm
    }

def save_gender_classifier(model: Any, output_path: str) -> None:
    """Guarda el pipeline completo de género en disco."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)