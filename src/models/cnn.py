from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

class SimpleCNN(nn.Module):
    """
    Input: [B, 1, N_MELS, T]
    Output: logits [B, num_classes]
    """
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, N_MELS, T]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 32, N_MELS/2, T/2]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, N_MELS/2, T/2]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, N_MELS/4, T/4]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # [B, 128, N_MELS/4, T/4]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, 1, N_MELS, T]
        x = self.features(x) # [B, 128, N_MELS/4, T/4]
        x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 128]
        x = self.dropout(x)
        return self.classifier(x)  # [B, num_classes]

class ResNet18(nn.Module):
    """
    Input: [B, 1, N_MELS, T]
    Output: logits [B, num_classes]
    """
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.resnet(x)  # [B, num_classes]
