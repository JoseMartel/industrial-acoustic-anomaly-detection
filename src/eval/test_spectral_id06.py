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

def evaluate_spectral_cross_id(target_id="06"):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "models/unsupervised/spectral_loss_model.pth"
    
    # Rutas de datos ID 06
    path_normal = f"data/6_dB_fan/fan/id_{target_id}/normal/*.wav"
    path_abnormal = f"data/6_dB_fan/fan/id_{target_id}/abnormal/*.wav"
    
    files_normal = sorted(glob.glob(path_normal))
    files_abnormal = sorted(glob.glob(path_abnormal))
    
    if not files_normal or not files_abnormal:
        print(f"Error: No se encontraron archivos para el ID {target_id}")
        return

    print(f"--- Evaluación Cruzada (Spectral Loss): Machine ID {target_id} ---")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = CAE().to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    mean = checkpoint['global_mean']
    std = checkpoint['global_std']
    
    def get_scores(files):
        ds = MIMIIDataset(files, global_mean=mean, global_std=std)
        ld = DataLoader(ds, batch_size=1, shuffle=False)
        scores = []
        with torch.no_grad():
            for batch in ld:
                batch = batch.to(DEVICE)
                output = model(batch)
                mse_per_frame = torch.mean((batch - output)**2, dim=2).cpu().numpy().flatten()
                scores.append(np.percentile(mse_per_frame, 90))
        return scores

    print("Evaluando sonidos normales...")
    scores_norm = get_scores(files_normal)
    print("Evaluando sonidos anormales...")
    scores_abnorm = get_scores(files_abnormal)
    
    y_true = [0]*len(scores_norm) + [1]*len(scores_abnorm)
    y_scores = scores_norm + scores_abnorm
    auc = roc_auc_score(y_true, y_scores)
    p_auc = roc_auc_score(y_true, y_scores, max_fpr=0.1)

    print(f"\n==============================================")
    print(f"RESULTADOS SPECTRAL LOSS PARA ID {target_id}")
    print(f"==============================================")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"p-AUC:   {p_auc:.4f}")
    print(f"==============================================")

if __name__ == "__main__":
    evaluate_spectral_cross_id("06")
