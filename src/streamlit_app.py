from __future__ import annotations

from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import joblib

from src.inference import predict_face

# definir rutas de los modelos
suggested_gender_model = Path("artifacts/models/pipeline_genero.pkl")
suggested_age_model = Path("artifacts/models/pipeline_edad.pkl")

@st.cache_resource
def cargar_modelos():
    try:
        gender_model = joblib.load(suggested_gender_model)
        age_model = joblib.load(suggested_age_model)
        return gender_model, age_model
    except FileNotFoundError:
        st.error("No se encontraron los modelos")
        return None, None


def run_app() -> None:
    """Ejecucion app visual"""

    st.set_page_config(page_title="Lab02 ML - Demo visual", layout="centered")
    st.title("Laboratorio 02: Deteccion e inferencia visual")
    st.write(
        "Esta app detecta rostros en una imagen y predice genero y edad usando modelos entrenados en el laboratorio."
    )
    # cargar modelos (aunque no se usen aun)
    gender_model, age_model = cargar_modelos()

    if gender_model is None or age_model is None:
        st.warning(
            "Los modelos no se cargaron correctamente. Asegúrese de ejecutar main.py primero."
        )
        st.stop()

    # subida de imagen
    uploaded_file = st.file_uploader(
        "Sube una fotografia, idealmente con rostros visibles",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.stop()

    # procesar imagen con OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # type: ignore

    # deteccion de rostros con Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # type: ignore

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # iterar sobre los rostros detectados
    for (x, y, w, h) in faces:
        # Extraemos el rostro original
        face_roi = image[y:y+h, x:x+w]
        
        # Enviar el rostro a nuestro pipeline modularizado
        gender_label, pred_age = predict_face(face_roi, gender_model, age_model)

        label = f"{gender_label}, Edad: {pred_age} anios"

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) #type: ignore
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) #type: ignore

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #type: ignore
    st.image(image_rgb, caption=f"Se detectaron {len(faces)} rostros", use_container_width=True)
    st.success("Inferencia completada")

if __name__ == "__main__":
    run_app()