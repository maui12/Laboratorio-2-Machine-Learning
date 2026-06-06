from _future_ import annotations

import numpy as np
import joblib
from dataclasses import dataclass
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(slots=True)
class AgeRegressionGuide:
    """Resume los pasos que deben seguir los estudiantes para la regresion."""

    scoring: str = "neg_mean_absolute_error"
    suggested_metrics: tuple[str, ...] = ("MAE", "RMSE", "R2")

    def to_text(self) -> str:
        """Genera un recordatorio corto para acompanar el laboratorio."""

        return (
            "Guia de regresion de edad\n"
            "========================\n"
            "\n"
            "La parte de regresion no se implementa en esta version.\n"
            "Los estudiantes deben completar src/regression.py siguiendo estos pasos:\n"
            "\n"
            "1. Reutilizar la misma matriz X preprocesada.\n"
            "2. Construir un Pipeline con PCA + LinearRegression.\n"
            "3. Ajustar pca__n_components con GridSearchCV.\n"
            "4. Evaluar con MAE, RMSE y R2.\n"
            "5. Guardar el modelo cuando este listo.\n"
        )


def build_age_regression_pipeline(random_state: int) -> Any:
    """Interfaz guia para la regresion de edad.

    TODO(estudiantes):
    - Importar PCA, LinearRegression y Pipeline.
    - Construir un pipeline equivalente al de clasificacion:

        Pipeline([
            ("pca", PCA(whiten=True, random_state=random_state)),
            ("reg", LinearRegression()),
        ])

    - Mantener la misma idea metodologica usada en el clasificador.
    """

    #completacion laboratorio
    pipeline = Pipeline([
        ("pca", PCA(whiten=True, random_state=random_state)),
        ("reg", LinearRegression()),
    ])

    return pipeline


def train_age_regressor(
    X_train: Any,
    y_age_train: Any,
    pca_components: tuple[int, ...],
    random_state: int,
) -> Any:
    """Interfaz guia para ajustar el regresor de edad.

    TODO(estudiantes):
    - Crear el pipeline con build_age_regression_pipeline.
    - Configurar GridSearchCV usando pca__n_components.
    - Sugerencia de scoring: neg_mean_absolute_error.
    - Retornar el mejor estimador encontrado.
    """
    #completacion laboratorio
    pipeline = build_age_regression_pipeline(random_state)
    param_grid = {
        "pca__n_components": pca_components,
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5, #5 particiones para validacion cruzada
        n_jobs=-1, #usa todos los nucleos del procesador 
    )
    grid_search.fit(X_train, y_age_train)
    return grid_search.best_estimator_


def evaluate_age_regressor(model: Any, X_test: Any, y_age_test: Any) -> dict[str, float]:
    """Interfaz guia para calcular metricas de regresion.

    TODO(estudiantes):
    - Obtener las predicciones con model.predict(X_test).
    - Calcular MAE.
    - Calcular RMSE.
    - Calcular R2.
    - Retornar un diccionario con esas metricas.
    """
    #completacion laboratorio
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_age_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_age_test, y_pred)))
    r2 = r2_score(y_age_test, y_pred)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }

def save_age_regressor(model: Any, output_path: str) -> None:
    """Interfaz guia para guardar el modelo de edad.

    TODO(estudiantes):
    - Importar joblib.
    - Guardar el pipeline completo, no solo el regresor final.
    - Usar un nombre sugerido como pipeline_edad.pkl.
    """
    #completacion laboratorio
    joblib.dump(model, output_path)