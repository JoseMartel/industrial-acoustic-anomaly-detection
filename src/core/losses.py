import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSpectralLoss(nn.Module):
    """
    Multi-Scale Spectral Loss para reconstrucción de espectrogramas.
    Combina MSE con una pérdida de convergencia espectral y magnitud logarítmica.
    """
    def __init__(self, alpha=0.5):
        super(MultiScaleSpectralLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, x, x_hat):
        # 1. MSE Estándar 
        mse_loss = self.mse(x, x_hat)
        
        # 2. Pérdida de Convergencia Espectral
        # Mide la diferencia en la forma de los picos espectrales
        norm_diff = torch.norm(x - x_hat, p='fro', dim=(2, 3))
        norm_orig = torch.norm(x, p='fro', dim=(2, 3))
        spectral_convergence = torch.mean(norm_diff / (norm_orig + 1e-6))
        
        # 3. Log-Magnitude Loss
        # Ayuda a reconstruir texturas finas
        log_mag_loss = F.l1_loss(x, x_hat)
        
        # Combinación ponderada
        total_loss = mse_loss + self.alpha * (spectral_convergence + log_mag_loss)
        
        return total_loss
