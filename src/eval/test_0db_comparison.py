import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# Importar componentes del núcleo
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.dataset import MIMIIDataset
from core.model_ae import CAE
from supervised.exp01_baseline_classifier.model_clf import CNNClassifier

def evaluate_0db():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_IDS = ["00", "02", "04"]
    
    # Rutas de modelos entrenados a 6dB
    UNSUPERVISED_MODEL = "models/unsupervised/best_model.pth"
    SUPERVISED_MODEL = "models/supervised/best_classifier.pth"
    
    # Recolectar archivos 0dB y tomar una submuestra para rapidez
    files_normal = []
    files_abnormal = []
    for machine_id in TARGET_IDS:
        norm = glob.glob(f"data/0_dB_fan/fan/id_{machine_id}/normal/*.wav")
        abnorm = glob.glob(f"data/0_dB_fan/fan/id_{machine_id}/abnormal/*.wav")
        files_normal.extend(norm[:200]) # 200 por ID
        files_abnormal.extend(abnorm[:200]) # 200 por ID
    
    if not files_normal:
        print("Error: No se encontraron archivos de 0dB.")
        return

    print(f"--- Evaluación en 0dB (Submuestra rápida) ---")
    print(f"IDs: {TARGET_IDS} | Normales: {len(files_normal)} | Anormales: {len(files_abnormal)}")

    # 1. EVALUAR CAE (No Supervisado)
    print("\n[1/2] Evaluando CAE (Entrenado en 6dB)...")
    checkpoint_ae = torch.load(UNSUPERVISED_MODEL, map_location=DEVICE, weights_only=False)
    model_ae = CAE().to(DEVICE)
    model_ae.load_state_dict(checkpoint_ae['model_state'])
    model_ae.eval()
    
    def get_ae_scores(files):
        ds = MIMIIDataset(files, global_mean=checkpoint_ae['global_mean'], global_std=checkpoint_ae['global_std'])
        ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
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

    # 2. EVALUAR CNN (Supervisado)
    print("[2/2] Evaluando CNN (Entrenada en 6dB)...")
    checkpoint_clf = torch.load(SUPERVISED_MODEL, map_location=DEVICE, weights_only=False)
    model_clf = CNNClassifier().to(DEVICE)
    model_clf.load_state_dict(checkpoint_clf['model_state'])
    model_clf.eval()
    
    def get_clf_scores(files):
        ds = MIMIIDataset(files, global_mean=checkpoint_clf['global_mean'], global_std=checkpoint_clf['global_std'])
        ld = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
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

    print(f"\n==============================================")
    print(f"RESULTADOS FINALES EN 0dB")
    print(f"==============================================")
    print(f"MODELO CAE (No Supervisado):")
    print(f"  ROC-AUC (0dB): {auc_ae:.4f}  (vs ~0.81 en 6dB)")
    print(f"----------------------------------------------")
    print(f"MODELO CNN (Supervisado):")
    print(f"  ROC-AUC (0dB): {auc_clf:.4f}  (vs 1.00 en 6dB)")
    print(f"==============================================")

if __name__ == "__main__":
    evaluate_0db()
