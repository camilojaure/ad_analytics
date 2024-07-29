import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import cv2
import torch
from ultralytics import YOLO
from pymongo import MongoClient
from tqdm import tqdm
from collections import defaultdict
import json

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración de MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['ad_analytics']
collection = db['ad_stats']

def detect_objects(keyframe_path, model):
    frame = cv2.imread(keyframe_path)
    if frame is None:
        logging.error(f"Error al leer el frame: {keyframe_path}")
        return None

    # Realizar la detección de objetos
    results = model(keyframe_path)
    
    # Extraer información de los objetos detectados
    detections = results[0].tojson()
    detections = json.loads(detections)
    
    return detections

def update_ad_features_in_db(video_name, ad_features):
    collection.update_one(
        {"video_name": video_name},
        {"$set": {"video_unique_detections": list(ad_features)}},
        upsert=True
    )
    logging.info(f"Características del anuncio actualizadas en la base de datos para {video_name}")

def main():
    keyframe_dir = "/Users/camilojaureguiberry/Documents/Projects/Research/ad_analytics/preprocessing/keyframes"

    # Cargar el modelo YOLOv8
    model = YOLO('yolov8s.pt')
    logging.info(f"Modelo YOLOv8 cargado con éxito")

    if not os.path.exists(keyframe_dir):
        logging.error(f"El directorio de keyframes no existe: {keyframe_dir}")
        return

    logging.info("Iniciando la detección de objetos y análisis de segmentos para todos los keyframes.")

    keyframe_files = [f for f in os.listdir(keyframe_dir) if f.endswith('.png')]
    if not keyframe_files:
        logging.error("No se encontraron keyframes en el directorio.")
        return

    unique_objects_per_video = defaultdict(set)

    for keyframe_file in tqdm(keyframe_files, desc="Procesando keyframes"):
        keyframe_path = os.path.join(keyframe_dir, keyframe_file)
        logging.info(f"Procesando keyframe: {keyframe_path}")

        if not os.path.isfile(keyframe_path):
            logging.error(f"El archivo no existe: {keyframe_path}")
            continue

        video_name, _, frame_num, _, cluster = keyframe_file.rsplit('_', 4)

        detections = detect_objects(keyframe_path, model)
        if detections is not None:
            for feature in detections:
                if feature["confidence"] > 0.5:
                    unique_objects_per_video[video_name].add(feature["name"])

    # Persistir los objetos únicos en la base de datos
    for video_name, unique_objects in unique_objects_per_video.items():
        update_ad_features_in_db(video_name, unique_objects)

if __name__ == "__main__":
    main()