import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# Importar componentes del núcleo (core)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.dataset import MIMIIDataset
from core.model_ae import CAE

def evaluate():
    # 1. Configuración de rutas (Relativas a la raíz del proyecto)
    TARGET_IDS = ["00", "02", "04"]
    MODEL_PATH = "models/unsupervised/best_model.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Cargar rutas
    val_files = []
    abnormal_files = []
    
    for machine_id in TARGET_IDS:
        path_normal = f"data/6_dB_fan/fan/id_{machine_id}/normal/*.wav"
        path_abnormal = f"data/6_dB_fan/fan/id_{machine_id}/abnormal/*.wav"
        
        all_normal = sorted(glob.glob(path_normal))
        if not all_normal:
            print(f"Advertencia: No se encontraron archivos normales para ID {machine_id}")
            continue
            
        # Usamos el mismo split que en el entrenamiento
        _, id_val_files = train_test_split(all_normal, test_size=0.2, random_state=42)
        val_files.extend(id_val_files)
        
        id_abnormal_files = sorted(glob.glob(path_abnormal))
        abnormal_files.extend(id_abnormal_files)
        print(f"ID {machine_id}: {len(id_val_files)} archivos de test (normal) y {len(id_abnormal_files)} anormales.")

    if not val_files or not abnormal_files:
        print("Error: No hay suficientes archivos para evaluar.")
        return

    # 3. Cargar Modelo y Metadatos Z-Score
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    global_mean = checkpoint.get('global_mean')
    global_std = checkpoint.get('global_std')
    
    model = CAE().to(DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    def get_scores(file_list):
        dataset = MIMIIDataset(file_list, global_mean=global_mean, global_std=global_std)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        scores = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                output = model(batch)
                
                # MSE por cada frame temporal
                mse_per_frame = torch.mean((batch - output)**2, dim=2).cpu().numpy().flatten()
                
                # Usamos Percentil 90
                score = np.percentile(mse_per_frame, 90)
                scores.append(score)
        return scores

    print("Evaluando sonidos normales...")
    normal_scores = get_scores(val_files)
    print("Evaluando sonidos anormales...")
    abnormal_scores = get_scores(abnormal_files)

    # 4. Métricas
    y_true = [0] * len(normal_scores) + [1] * len(abnormal_scores)
    y_scores = normal_scores + abnormal_scores

    auc = roc_auc_score(y_true, y_scores)
    p_auc = roc_auc_score(y_true, y_scores, max_fpr=0.1)

    print(f"\n--- RESULTADOS FINALES (Z-SCORE + PERCENTILE 90) ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"p-AUC (max_fpr=0.1): {p_auc:.4f}")

    # --- Ejemplo Visual de Reconstrucción ---
    os.makedirs("results/figures", exist_ok=True)
    sample_dataset = MIMIIDataset([abnormal_files[0]], global_mean=global_mean, global_std=global_std)
    sample_input = sample_dataset[0].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sample_output = model(sample_input)
    
    orig = sample_input.squeeze().cpu().numpy()
    rec = sample_output.squeeze().cpu().numpy()
    err = np.abs(orig - rec)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(orig, aspect='auto', origin='lower')
    plt.title("Original (Abnormal)")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(rec, aspect='auto', origin='lower')
    plt.title("Reconstrucción")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(err, aspect='auto', origin='lower')
    plt.title("Error (Diferencia)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("results/figures/reconstruction_example.png")
    
    # 5. Guardar Gráfica de Distribución
    plt.figure(figsize=(10, 5))
    plt.hist(normal_scores, bins=50, alpha=0.5, label='Normal', color='blue', density=True)
    plt.hist(abnormal_scores, bins=50, alpha=0.5, label='Abnormal', color='red', density=True)
    plt.title(f"Distribución de Errores (Z-Score + Percentil 90)\nAUC: {auc:.4f}")
    plt.legend()
    plt.savefig("results/figures/results_evaluation.png")
    
    print("\nResultados guardados en results/figures/")

if __name__ == "__main__":
    evaluate()
