import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Importar componentes del núcleo y locales
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.dataset import MIMIIDataset
from model_clf import CNNClassifier

def evaluate():
    # 1. Configuración
    TARGET_IDS = ["00", "02", "04"]
    MODEL_PATH = "models/supervised/best_classifier.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Recolección de archivos (Mismo split aleatorio que en train)
    all_files = []
    all_labels = []

    for machine_id in TARGET_IDS:
        path_normal = f"data/6_dB_fan/fan/id_{machine_id}/normal/*.wav"
        path_abnormal = f"data/6_dB_fan/fan/id_{machine_id}/abnormal/*.wav"
        
        norm_files = sorted(glob.glob(path_normal))
        abnorm_files = sorted(glob.glob(path_abnormal))
        
        all_files.extend(norm_files)
        all_labels.extend([0] * len(norm_files))
        
        all_files.extend(abnorm_files)
        all_labels.extend([1] * len(abnorm_files))

    _, val_files, _, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # 3. Cargar Modelo y Metadatos
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CNNClassifier().to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    dataset = MIMIIDataset(
        val_files, 
        labels=val_labels, 
        global_mean=checkpoint['global_mean'], 
        global_std=checkpoint['global_std']
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 4. Inferencia
    y_true = []
    y_scores = []
    
    print("Iniciando evaluación del clasificador supervisado...")
    with torch.no_grad():
        for batch, label in loader:
            batch = batch.to(DEVICE)
            output = model(batch)
            y_scores.append(output.item())
            y_true.append(label.item())

    # 5. Métricas
    auc = roc_auc_score(y_true, y_scores)
    p_auc = roc_auc_score(y_true, y_scores, max_fpr=0.1)

    print(f"\n--- RESULTADOS CLASIFICADOR SUPERVISADO ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"p-AUC (max_fpr=0.1): {p_auc:.4f}")

if __name__ == "__main__":
    evaluate()
