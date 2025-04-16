import torch
import torch.nn as nn
import torch.nn.functional as F

# --- PSF-prediktor hálózat ---
class PSFPredictor(nn.Module):
    def __init__(self, kernel_size=15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, kernel_size * kernel_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, 1, self.kernel_size, self.kernel_size)
        x = F.relu(x)
        x = x / (x.sum(dim=[2, 3], keepdim=True) + 1e-8)
        return x