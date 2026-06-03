import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Importar componentes del núcleo y locales
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.dataset import MIMIIDataset
from model_clf import CNNClassifier

def train():
    # 1. Configuración
    TARGET_IDS = ["00", "02", "04"]
    BATCH_SIZE = 32
    LR = 1e-4 # Un poco más bajo para estabilidad en clasificador
    EPOCHS = 30
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PATH = "models/supervised/best_classifier.pth"
    os.makedirs("models/supervised", exist_ok=True)

    # 2. Recolección de archivos y etiquetas
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

    # Split: 80% train, 20% test (Comparación justa: mismo split aleatorio)
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # 3. Calcular Z-Score Global (Solo con datos de entrenamiento normales para ser justos)
    print("Calculando estadísticas globales (Z-Score)...")
    train_norm_files = [f for f, l in zip(train_files, train_labels) if l == 0]
    temp_dataset = MIMIIDataset(train_norm_files, normalize=False)
    
    sums = np.zeros((128, 1))
    sq_sums = np.zeros((128, 1))
    total_frames = 0
    
    for i in range(min(200, len(train_norm_files))): # Muestreo de 200 archivos
        spec = temp_dataset[i].squeeze().numpy()
        sums += np.mean(spec, axis=1, keepdims=True)
        sq_sums += np.mean(spec**2, axis=1, keepdims=True)
        total_frames += 1
        
    global_mean = sums / total_frames
    global_std = np.sqrt(sq_sums / total_frames - global_mean**2)

    # 4. DataLoaders
    train_dataset = MIMIIDataset(train_files, labels=train_labels, global_mean=global_mean, global_std=global_std)
    val_dataset = MIMIIDataset(val_files, labels=val_labels, global_mean=global_mean, global_std=global_std)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Modelo, Loss y Optimizer
    model = CNNClassifier().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 6. Loop de Entrenamiento
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch, labels in train_loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validación
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, labels in val_loader:
                batch, labels = batch.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(batch)
                val_loss += criterion(outputs, labels).item()
                
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        acc = correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} Acc: {acc:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'model_state': model.state_dict(),
                'global_mean': global_mean,
                'global_std': global_std
            }, SAVE_PATH)
            print(f"--> Modelo guardado en {SAVE_PATH}")

if __name__ == "__main__":
    train()
