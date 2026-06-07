import os
import subprocess
import sys
from sklearn.model_selection import train_test_split
from src.config import LabConfig
from src.data import build_dataset, dataset_to_dataframe
from src.gender_classification import train_gender_classifier, evaluate_gender_classifier, save_gender_classifier
from src.regression import train_age_regressor, evaluate_age_regressor, save_age_regressor
from src.visualization import plot_real_vs_predicted_age

def main():
    print("--- FASE 1: PREPARACIÓN DE DATOS ---")
    
    # 1. Cargamos la configuración
    config = LabConfig()
    print(f"Buscando imágenes en: {config.dataset_dir} (ruta configurable en config.py)")
    
    # 2. Construimos el dataset
    print("Construyendo matriz X e y (esto puede tomar un tiempo)...")
    dataset = build_dataset(config)
    
    # 3. Verificamos los resultados
    print(f"¡Dataset construido con {len(dataset)} imágenes válidas!")
    print(f"Clasificación extraida de los nombres de archivos de UTKFace")
    if dataset.skipped_files:
        print(f"Se omitieron {len(dataset.skipped_files)} archivos por error de formato.")
        
    # 4. Usamos pandas para ver un resumen (Data Understanding)
    df = dataset_to_dataframe(dataset)
    print("\nPrimeras 10 filas del dataset:")
    print(df.head(10))
    
    print("\nDistribución de Género (0: Mujer, 1: Hombre):")
    print(df["gender"].value_counts())


    print("\n--- SEPARACIÓN DE DATOS (TRAIN/TEST SPLIT) ---")
    # Separamos X, el género y la edad simultáneamente en un 80% entrenamiento y 20% prueba
    X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(
        dataset.X,
        dataset.y_gender,
        dataset.y_age,
        test_size=config.test_size,
        random_state=config.random_state
    )
    print(f"Datos de entrenamiento: {X_train.shape[0]} imágenes")
    print(f"Datos de prueba: {X_test.shape[0]} imágenes")


    print("--- FASE 2: MODELADO Y EVALUACIÓN ---")
    # --- ENTRENAMIENTO DE GÉNERO ---
    print("\nEntrenando pipeline de clasificación de género (PCA + GaussianNB)...")
    pca_components_gender_to_try = [15, 25, 35, 50] 
    
    gender_model = train_gender_classifier(
        X_train, 
        y_gender_train, 
        pca_components_gender_to_try, 
        config.random_state
    )
    
    print("\nEvaluando modelo de género...")
    gender_metrics = evaluate_gender_classifier(gender_model, X_test, y_gender_test)
    print("      Exactitud (Accuracy):  {:.4f}".format(gender_metrics['accuracy']))
    print("      Precisión (Precision): {:.4f}".format(gender_metrics['precision']))
    print("      Exhaustividad (Recall):{:.4f}".format(gender_metrics['recall']))
    print("      Puntaje F1 (F1-score): {:.4f}".format(gender_metrics['f1']))
    print("      Matriz de Confusión:")
    print("                 Pred: Mujer   Pred: Hombre")
    print(f"      Real Mujer:   {gender_metrics['confusion_matrix'][0][0]:<12} {gender_metrics['confusion_matrix'][0][1]}")
    print(f"      Real Hombre:  {gender_metrics['confusion_matrix'][1][0]:<12} {gender_metrics['confusion_matrix'][1][1]}")
    
    os.makedirs("artifacts/models", exist_ok=True)
    
    save_gender_classifier(gender_model, "artifacts/models/pipeline_genero.pkl")
    print("¡pipeline_genero.pkl guardado exitosamente!")


    print("\n--- ENTRENAMIENTO DE REGRESIÓN DE EDAD ---")
    print("Buscando los mejores hiperparámetros con GridSearchCV...")

    # El equipo define qué cantidades de componentes PCA probar
    pca_components_to_try = (50, 100, 150) 

    # 1. Entrenar (Esto ejecutará GridSearchCV internamente)
    best_age_model = train_age_regressor(
        X_train=X_train, 
        y_age_train=y_age_train, 
        pca_components=pca_components_to_try, 
        random_state=config.random_state
    )

    # 2. Evaluar y mostrar métricas numéricas puras en consola
    print("\nEvaluando el mejor modelo de edad...")
    age_metrics = evaluate_age_regressor(best_age_model, X_test, y_age_test)

    for metric, value in age_metrics.items():
        print(f"- {metric}: {value:.4f}")

    print("Generando gráfico de Edad Real vs Predicha...")
    y_pred_age = best_age_model.predict(X_test)
    plot_real_vs_predicted_age(y_age_test, y_pred_age)
    print("      -> Gráfico guardado en artifacts/real_vs_pred_age.png")

    # 3. Guardar el artefacto para la app web
    os.makedirs("artifacts/models", exist_ok=True)
    save_age_regressor(best_age_model, "artifacts/models/pipeline_edad.pkl")
    print("¡pipeline_edad.pkl guardado exitosamente!")


    print("\n--- FASE 3: DESPLIEGUE ---")
    print("Iniciando la aplicación visual en tu navegador...")
    print("Presiona Ctrl+C en esta consola para detener el servidor web cuando termines.\n")
    
    # Esto equivale a escribir "streamlit run src/streamlit_app.py" en la terminal
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py"])

if __name__ == "__main__":
    main()