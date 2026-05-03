import torch
import torch.nn as nn

class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        
        # --- ENCODER ---
        # Entrada: (1, 128, 313)
        self.encoder = nn.Sequential(
            # Capa 1: Reducción a (16, 64, 157)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Capa 2: Reducción a (32, 32, 79)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Capa 3: Reducción a (64, 16, 40)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Capa 4: Reducción a (128, 8, 20)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.2) # Evita memorizar ruido
        )
        
        # --- DECODER ---
        self.decoder = nn.Sequential(
            # Capa 1: De (128, 8, 20) a (64, 16, 40)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Capa 2: De (64, 16, 40) a (32, 32, 80)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Capa 3: De (32, 32, 80) a (16, 64, 160)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Capa 4 final: De (16, 64, 160) a (1, 128, 320)
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
            # Sin Sigmoid para permitir valores negativos de la normalización Z-score
        )

    def forward(self, x):
        # Guardar dimensiones de entrada para recorte final si es necesario
        batch_size, channels, h, w = x.size()
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        # Recorte (Crop) para asegurar que la salida sea idéntica a la entrada (128, 313)
        # Esto es necesario debido a que las dimensiones impares (313) no son divisibles exactas
        return x[:, :, :h, :w]

if __name__ == "__main__":
    # --- PRUEBA DE ARQUITECTURA ---
    model = CAE()
    
    # Simular un batch de entrada (BatchSize=2, Canal=1, Mels=128, Frames=313)
    input_tensor = torch.randn(2, 1, 128, 313)
    
    # Pasar por el modelo
    output_tensor = model(input_tensor)
    
    print("Verificación de Dimensiones del Modelo:")
    print(f"Entrada: {input_tensor.shape}")
    print(f"Salida:  {output_tensor.shape}")
    
    if input_tensor.shape == output_tensor.shape:
        print("\n¡Éxito! La arquitectura reconstruye las dimensiones correctamente.")
    else:
        print("\nError: Las dimensiones no coinciden.")
