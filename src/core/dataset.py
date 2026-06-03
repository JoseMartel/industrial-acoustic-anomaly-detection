import os
import glob
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MIMIIDataset(Dataset):
    """
    Dataset class con Normalización Z-Score por Frecuencia.
    Soporta etiquetas opcionales para entrenamiento supervisado.
    """
    def __init__(self, file_paths, labels=None, n_mels=128, n_fft=1024, hop_length=512, global_mean=None, global_std=None, normalize=True):
        self.file_paths = file_paths
        self.labels = labels
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.global_mean = global_mean # Debe ser shape (128, 1)
        self.global_std = global_std   # Debe ser shape (128, 1)
        self.normalize = normalize

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # 1. Cargar audio
        y, sr = librosa.load(file_path, sr=None)
        
        # 2. Generar Espectrograma Mel
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        # 3. Convertir a Log-Mel
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # 4. Normalización Z-Score por Frecuencia
        if self.normalize:
            if self.global_mean is not None and self.global_std is not None:
                # Broadcasting automático: (128, 313) - (128, 1) / (128, 1)
                normalized_spec = (log_mel_spec - self.global_mean) / (self.global_std + 1e-6)
            else:
                # Fallback: Normalización por frecuencia local si no hay globales
                m = np.mean(log_mel_spec, axis=1, keepdims=True)
                s = np.std(log_mel_spec, axis=1, keepdims=True)
                normalized_spec = (log_mel_spec - m) / (s + 1e-6)
        else:
            normalized_spec = log_mel_spec
        
        # 5. Convertir a Tensor
        tensor_spec = torch.from_numpy(normalized_spec).unsqueeze(0).float()
        
        # 6. Retornar con etiqueta si existe
        if self.labels is not None:
            label = torch.tensor(self.labels[idx]).float()
            return tensor_spec, label
            
        return tensor_spec
