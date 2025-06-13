import torch
import torch.nn as nn

# === 1. Простая сеть для 2D бинарной классификации ===
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.RelU(),
            nn.Linear(16,1),
            nn.Sigmoid() # для ВСЕLoss
        )

    def forward(self, x):
        return self.net(x)