"""
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████

$$\   $$\                    $$$$$$$$\                                         $$$$$$$$\             $$\                                    $$\                         
$$ | $$  |                   $$  _____|                                        $$  _____|            $$ |                                   $$ |                        
$$ |$$  / $$$$$$\  $$\   $$\ $$ |    $$$$$$\  $$$$$$\  $$$$$$\$$$$\   $$$$$$\  $$ |      $$\   $$\ $$$$$$\    $$$$$$\  $$$$$$\   $$$$$$$\ $$$$$$\    $$$$$$\   $$$$$$\  
$$$$$  / $$  __$$\ $$ |  $$ |$$$$$\ $$  __$$\ \____$$\ $$  _$$  _$$\ $$  __$$\ $$$$$\    \$$\ $$  |\_$$  _|  $$  __$$\ \____$$\ $$  _____|\_$$  _|  $$  __$$\ $$  __$$\ 
$$  $$<  $$$$$$$$ |$$ |  $$ |$$  __|$$ |  \__|$$$$$$$ |$$ / $$ / $$ |$$$$$$$$ |$$  __|    \$$$$  /   $$ |    $$ |  \__|$$$$$$$ |$$ /        $$ |    $$ /  $$ |$$ |  \__|
$$ |\$$\ $$   ____|$$ |  $$ |$$ |   $$ |     $$  __$$ |$$ | $$ | $$ |$$   ____|$$ |       $$  $$<    $$ |$$\ $$ |     $$  __$$ |$$ |        $$ |$$\ $$ |  $$ |$$ |      
$$ | \$$\\$$$$$$$\ \$$$$$$$ |$$ |   $$ |     \$$$$$$$ |$$ | $$ | $$ |\$$$$$$$\ $$$$$$$$\ $$  /\$$\   \$$$$  |$$ |     \$$$$$$$ |\$$$$$$$\   \$$$$  |\$$$$$$  |$$ |      
\__|  \__|\_______| \____$$ |\__|   \__|      \_______|\__| \__| \__| \_______|\________|\__/  \__|   \____/ \__|      \_______| \_______|   \____/  \______/ \__|      
                   $$\   $$ |                                                                                                                                           
                   \$$$$$$  |                                                                                                                                           
                    \______/                                                                                                                                            

	1.	Extracción de Frames:
▓▓ Utiliza FFmpeg para extraer frames de los videos.
▓▓ Los frames extraídos se guardan en un directorio especificado.

	2.	Clustering con KMeans:
▓▓ Los frames extraídos se redimensionan a 64x64 píxeles para reducir la complejidad computacional.
▓▓ Se convierten en vectores unidimensionales.
▓▓ El número óptimo de clusters se determina utilizando el silhouette score.
▓▓ Los frames se agrupan en clusters utilizando KMeans con el número óptimo de clusters.

	3.	Selección de Frames Representativos:
▓▓ Se selecciona el frame más cercano al centroide de cada cluster.
▓▓ Los frames seleccionados se guardan en el directorio keyframes con nombres que referencian al video original.

████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████

"""
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import logging
import subprocess

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_frames(video_path, frame_dir, fps=5):
    """Extrae frames de un video usando FFmpeg."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_output_pattern = os.path.join(frame_dir, f"{video_name}_frame_%04d.png")
    
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    cmd = [
        'ffmpeg', '-i', video_path, '-vf', f'fps={fps}', frame_output_pattern,
        '-hide_banner', '-loglevel', 'error'
    ]
    subprocess.run(cmd, check=True)
    logging.info(f"Frames extraídos para el video: {video_name}")

def read_frames(frame_dir, video_name):
    """Lee frames del directorio especificado."""
    frame_files = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.startswith(video_name) and f.endswith('.png')]
    frame_files.sort()
    
    frames = []
    for file in frame_files:
        frame = cv2.imread(file)
        if frame is not None:
            frames.append(frame)
    
    logging.info(f"Se leyeron {len(frames)} frames del video: {video_name}")
    return frames, frame_files

def extract_features(frames):
    """Extrae características de los frames para el clustering."""
    features = []
    for frame in frames:
        # Redimensionar frame para análisis rápido
        resized_frame = cv2.resize(frame, (64, 64)).flatten()
        
        # Calcular histograma de color
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Combinar características
        combined_features = np.hstack((resized_frame, hist))
        features.append(combined_features)
    
    return features

def find_optimal_clusters(data, min_clusters, max_clusters):
    """Encuentra el número óptimo de clusters usando el silhouette score."""
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append((n_clusters, score))
        logging.info(f"Silueta score para {n_clusters} clusters: {score:.4f}")
    
    optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    logging.info(f"Número óptimo de clusters encontrado: {optimal_clusters}")
    return optimal_clusters

def extract_keyframes(video_name, frame_dir, keyframe_dir, min_clusters=4, max_clusters=15, frames_per_cluster=5):
    """Extrae keyframes de un video."""
    # Crear el directorio de keyframes si no existe
    if not os.path.exists(keyframe_dir):
        os.makedirs(keyframe_dir)

    # Leer frames
    frames, frame_files = read_frames(frame_dir, video_name)
    
    if len(frames) < min_clusters:
        logging.warning(f"No hay suficientes frames para agrupar en el video: {video_name}")
        return

    # Extraer características de los frames
    features = extract_features(frames)
    
    # Encontrar el número óptimo de clusters
    optimal_clusters = find_optimal_clusters(features, min_clusters=min_clusters, max_clusters=max_clusters)
    
    # Usar KMeans para agrupar frames en clusters
    kmeans = KMeans(n_clusters=optimal_clusters, n_init='auto')
    kmeans.fit(features)
    
    # Seleccionar los frames más cercanos al centroide de cada cluster
    keyframes = []
    for cluster in range(optimal_clusters):
        cluster_indices = np.where(kmeans.labels_ == cluster)[0]
        if len(cluster_indices) > 0:
            cluster_center = kmeans.cluster_centers_[cluster]
            sorted_indices = sorted(cluster_indices, key=lambda index: np.linalg.norm(features[index] - cluster_center))
            selected_indices = sorted_indices[:frames_per_cluster]
            for idx in selected_indices:
                keyframes.append((idx, frames[idx], cluster))

    # Guardar frames representativos
    for i, (index, keyframe, cluster) in enumerate(keyframes):
        keyframe_filename = f'{video_name}_frame_{index:04d}_cluster_{cluster}.png'
        cv2.imwrite(os.path.join(keyframe_dir, keyframe_filename), keyframe)
        logging.info(f"Keyframe guardado: {keyframe_filename}")

def main():
    video_directory = "/Users/camilojaureguiberry/Library/Mobile Documents/com~apple~CloudDocs/MMA/DeepLearning/ad_analytics/videos"
    frame_dir = 'frames'
    keyframe_dir = 'keyframes'
    fps = 5

    video_paths = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith(('.mp4', '.mov', '.avi', '.mkv'))]

    logging.info("Iniciando la extracción de frames y keyframes para todos los videos.")
    
    for video_path in tqdm(video_paths, desc="Procesando videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        extract_frames(video_path, frame_dir, fps=fps)
        extract_keyframes(video_name, frame_dir, keyframe_dir, min_clusters=4, max_clusters=15, frames_per_cluster=1)

if __name__ == "__main__":
    main()