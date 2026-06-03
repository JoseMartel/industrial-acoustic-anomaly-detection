import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Importar componentes del núcleo (core)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.dataset import MIMIIDataset
from core.model_ae import CAE
from core.losses import MultiScaleSpectralLoss

def train():
    # 1. Configuración de rutas (Relativas a la raíz del proyecto)
    TARGET_IDS = ["00", "02", "04"]
    BATCH_SIZE = 32
    LR = 1e-3
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "models/unsupervised/spectral_loss_model.pth"

    print(f"Usando dispositivo: {DEVICE}")

    # 2. Preparar lista de archivos
    all_normal_files = []
    for machine_id in TARGET_IDS:
        path = f"data/6_dB_fan/fan/id_{machine_id}/normal/*.wav"
        files = sorted(glob.glob(path))
        print(f"ID {machine_id}: Encontrados {len(files)} archivos normales.")
        all_normal_files.extend(files)

    if not all_normal_files:
        print(f"Error: No se encontraron archivos para los IDs {TARGET_IDS}")
        return

    train_files, val_files = train_test_split(all_normal_files, test_size=0.2, random_state=42)
    
    # 3. Calcular Estadísticas por Frecuencia
    print(f"Calculando estadísticas por frecuencia sobre {len(train_files)} archivos...")
    temp_dataset = MIMIIDataset(train_files, normalize=False)
    all_specs = []
    for i in range(len(temp_dataset)):
        spec = temp_dataset[i]
        all_specs.append(spec.squeeze(0).numpy())
    
    all_specs_concat = np.concatenate(all_specs, axis=1)
    global_mean = np.mean(all_specs_concat, axis=1, keepdims=True)
    global_std = np.std(all_specs_concat, axis=1, keepdims=True)
    
    print(f"Estadísticas calculadas. Media promedio: {np.mean(global_mean):.4f}")

    # 4. Crear DataLoaders
    train_dataset = MIMIIDataset(train_files, global_mean=global_mean, global_std=global_std)
    val_dataset = MIMIIDataset(val_files, global_mean=global_mean, global_std=global_std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 5. Instanciar Modelo
    model = CAE().to(DEVICE)
    criterion = MultiScaleSpectralLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 6. Ciclo de Entrenamiento
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            outputs = model(batch)
            loss = criterion(outputs, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                outputs = model(batch)
                loss = criterion(outputs, batch)
                val_loss += loss.item() * batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        print(f"Época [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state = {
                'model_state': model.state_dict(),
                'global_mean': global_mean,
                'global_std': global_std
            }
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(state, MODEL_SAVE_PATH)
            print(f"--> Guardado Checkpoint: {MODEL_SAVE_PATH}")

    # Guardar gráfica de pérdida
    os.makedirs("results/figures", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.title('Curva de Aprendizaje (Z-Score por Frecuencia)')
    plt.legend()
    plt.savefig('results/figures/training_loss.png')
    
    print("\nEntrenamiento finalizado. Gráfica guardada en results/figures/training_loss.png")

if __name__ == "__main__":
    train()
