"""
models.py
CNN model definition for binary COVID-CT classification.

Label mapping:
    0 = normal
    1 = covid

The model returns one raw logit per image. Apply torch.sigmoid(logits)
to obtain COVID probabilities.
"""

import torch.nn as nn


class BaselineCNN(nn.Module):
    """Lightweight convolutional neural network trained from scratch."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()

        if num_classes != 1:
            raise ValueError("Binary BCE model requires num_classes=1.")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)
