import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        # --- ENCODER (Mismo que el CAE para comparación justa) ---
        # Entrada: (1, 128, 313)
        self.encoder = nn.Sequential(
            # Capa 1: (16, 64, 157)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            # Capa 2: (32, 32, 79)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Capa 3: (64, 16, 40)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Capa 4: (128, 8, 20)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.2)
        )
        
        # --- CABEZA CLASIFICADORA ---
        # Aplanamos el bottleneck de 128*8*20 = 20480
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 20, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid() # Salida binaria (0: Normal, 1: Anómalo)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = CNNClassifier()
    input_tensor = torch.randn(2, 1, 128, 313)
    output = model(input_tensor)
    print(f"Entrada: {input_tensor.shape}")
    print(f"Salida (Probabilidad): {output.shape}")
    print(output)
