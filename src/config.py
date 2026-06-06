from dataclasses import dataclass
from pathlib import Path

@dataclass(slots=True)
class LabConfig:
    #Ajusta esta ruta a donde realmente esta tu carpeta UTKFace
    dataset_dir: Path = Path("dataset/UTKFace") 
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    image_size: tuple[int, int] = (25, 25)
    max_images: int | None = 500 # cantidad de imagenes cargadas al inicio
    random_state: int = 42
    test_size: float = 0.2