#cnn_model.py
"""
Convolutional neural network architecture for attention‑guided
tabular‑to‑image representations.

This module defines the `TabNetCNN` class, a lightweight CNN designed to
process the 2D image representations produced by the deterministic layout
stage. The architecture is intentionally kept simple and fixed across all
experiments to ensure that any performance differences originate from the
layout geometry rather than from model capacity.

Architecture design:
  - Small branch (≤16 pixels): 8→16 channels, 2×2 convolutions.
  - Medium branch (≤100 pixels): 16→32 channels, 3×3 convolutions.
  - Default branch (>100 pixels): 32→64 channels, 3×3 convolutions.
  All branches end with adaptive average pooling to 1×1, followed by a
  dropout layer and a linear classifier.

The model requires the final image height and width to select the appropriate
branch, but the spatial dimensions are determined entirely by the chosen layout
strategy; the CNN itself does not impose any prior on feature organisation.

**Role in the Map–Optimize–Learn pipeline:**
  - **Learn:** The CNN is trained exclusively on the frozen image
    representations, with no gradient flow back to TabNet or the layout
    builder. This strict decoupling enables controlled experimentation
    with different spatial layouts while keeping the learner identical.
  - The architecture is shared across all tabular‑to‑image baselines
    (IGTD, naive reshape, AG‑T2I variants) to guarantee fair comparison.
"""
import torch.nn as nn

class TabNetCNN(nn.Module):
    """
    CNN for TabNet-generated image representations.

    Standard interface:
        n_classes
        input_channels (C)
        image_height (H)
        image_width (W)
        dropout (float) – dropout rate for the final FC layer
    """

    def __init__(self, n_classes, input_channels=1, image_height=None, image_width=None, dropout=0.3):
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width

        total_pixels = (
            image_height * image_width if image_height and image_width else None
        )

        # -------------------------
        # SMALL IMAGE BRANCH
        # -------------------------
        if total_pixels is not None and total_pixels <= 16:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 8, kernel_size=2, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),

                nn.Conv2d(8, 16, kernel_size=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),

                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc_input = 16

        # -------------------------
        # MEDIUM IMAGE BRANCH
        # -------------------------
        elif total_pixels is not None and total_pixels <= 100:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc_input = 32

        # -------------------------
        # DEFAULT (ROBUST)
        # -------------------------
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc_input = 64

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fc_input, n_classes)
        )

    def forward(self, x):
        # Handle possible [B,1,H,W] or [B,C,H,W]
        if x.dim() == 5:
            x = x.squeeze(1)

        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B,C,H,W], got {x.shape}")

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)