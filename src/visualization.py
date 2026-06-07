import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Any

def plot_real_vs_predicted_age(y_real: Any, y_pred: Any, output_path: str = "artifacts/real_vs_pred_age.png") -> None:
    """
    Genera y guarda un gráfico de dispersión comparando la edad real y la predicha.
    """
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_real, y_pred, alpha=0.4, color='royalblue', edgecolor='k', s=30)
    
    min_val = min(np.min(y_real), np.min(y_pred))
    max_val = max(np.max(y_real), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Predicción Perfecta')
    
    plt.title('Comparativa: Edad Real vs Edad Predicha', fontsize=14)
    plt.xlabel('Edad Real (Años)', fontsize=12)
    plt.ylabel('Edad Predicha por el Modelo (Años)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    # Crear directorio si no existe y guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()