from src.config import LabConfig
from src.data import build_dataset, dataset_to_dataframe

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

if __name__ == "__main__":
    main()