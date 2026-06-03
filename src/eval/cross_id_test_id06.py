import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Importar componentes del núcleo
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.dataset import MIMIIDataset
from core.model_ae import CAE
from supervised.exp01_baseline_classifier.model_clf import CNNClassifier

def cross_id_evaluation(target_id="06"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Rutas de modelos
    UNSUPERVISED_MODEL = "models/unsupervised/best_model.pth"
    SUPERVISED_MODEL = "models/supervised/best_classifier.pth"
    
    # Rutas de datos ID 06
    path_normal = f"data/6_dB_fan/fan/id_{target_id}/normal/*.wav"
    path_abnormal = f"data/6_dB_fan/fan/id_{target_id}/abnormal/*.wav"
    
    files_normal = sorted(glob.glob(path_normal))
    files_abnormal = sorted(glob.glob(path_abnormal))
    
    if not files_normal or not files_abnormal:
        print(f"Error: No se encontraron archivos para el ID {target_id}")
        return

    print(f"--- Evaluación Cruzada: Machine ID {target_id} (No visto en entrenamiento) ---")
    print(f"Normales: {len(files_normal)}, Anormales: {len(files_abnormal)}")

    # 1. EVALUAR MODELO NO SUPERVISADO (CAE)
    print("\n[1/2] Evaluando CAE (No Supervisado)...")
    checkpoint_ae = torch.load(UNSUPERVISED_MODEL, map_location=DEVICE, weights_only=False)
    model_ae = CAE().to(DEVICE)
    model_ae.load_state_dict(checkpoint_ae['model_state'])
    model_ae.eval()
    
    mean_ae = checkpoint_ae['global_mean']
    std_ae = checkpoint_ae['global_std']
    
    def get_ae_scores(files):
        ds = MIMIIDataset(files, global_mean=mean_ae, global_std=std_ae)
        ld = DataLoader(ds, batch_size=1, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in ld:
                batch = batch.to(DEVICE)
                output = model_ae(batch)
                mse_per_frame = torch.mean((batch - output)**2, dim=2).cpu().numpy().flatten()
                scores.append(np.percentile(mse_per_frame, 90))
        return scores

    scores_ae_norm = get_ae_scores(files_normal)
    scores_ae_abnorm = get_ae_scores(files_abnormal)
    
    y_true = [0]*len(scores_ae_norm) + [1]*len(scores_ae_abnorm)
    y_scores_ae = scores_ae_norm + scores_ae_abnorm
    auc_ae = roc_auc_score(y_true, y_scores_ae)
    p_auc_ae = roc_auc_score(y_true, y_scores_ae, max_fpr=0.1)

    # 2. EVALUAR MODELO SUPERVISADO (CNN)
    print("[2/2] Evaluando CNN (Supervisado)...")
    checkpoint_clf = torch.load(SUPERVISED_MODEL, map_location=DEVICE, weights_only=False)
    model_clf = CNNClassifier().to(DEVICE)
    model_clf.load_state_dict(checkpoint_clf['model_state'])
    model_clf.eval()
    
    mean_clf = checkpoint_clf['global_mean']
    std_clf = checkpoint_clf['global_std']
    
    def get_clf_scores(files):
        ds = MIMIIDataset(files, global_mean=mean_clf, global_std=std_clf)
        ld = DataLoader(ds, batch_size=1, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in ld:
                batch = batch.to(DEVICE)
                output = model_clf(batch)
                scores.append(output.item())
        return scores

    scores_clf_norm = get_clf_scores(files_normal)
    scores_clf_abnorm = get_clf_scores(files_abnormal)
    
    y_scores_clf = scores_clf_norm + scores_clf_abnorm
    auc_clf = roc_auc_score(y_true, y_scores_clf)
    p_auc_clf = roc_auc_score(y_true, y_scores_clf, max_fpr=0.1)

    # RESULTADOS FINALES
    print(f"\n==============================================")
    print(f"RESULTADOS FINALES PARA ID {target_id}")
    print(f"==============================================")
    print(f"MODELO CAE (No Supervisado):")
    print(f"  ROC-AUC: {auc_ae:.4f}")
    print(f"  p-AUC:   {p_auc_ae:.4f}")
    print(f"----------------------------------------------")
    print(f"MODELO CNN (Supervisado):")
    print(f"  ROC-AUC: {auc_clf:.4f}")
    print(f"  p-AUC:   {p_auc_clf:.4f}")
    print(f"==============================================")

if __name__ == "__main__":
    cross_id_evaluation("06")
