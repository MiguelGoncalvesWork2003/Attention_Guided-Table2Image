# cnn_architectures.py
"""
Architecture registry for the layout-reuse experiment (E1).

Purpose
-------
The AGT2I claim under test is that a feature-to-pixel layout, produced without
any reference to the downstream classifier, retains its value when that
classifier changes. Testing this requires holding the *images* fixed and
varying only the network that consumes them.

Every architecture here therefore obeys the same contract as the existing
`TabNetCNN`:

    Model(n_classes, input_channels, image_height, image_width, dropout)

so that `train_cnn.py` and `evaluate_cnn.py` can swap between them through a
single environment variable, reading exactly the same `X_*_img.npy` files
produced once by `tabnet_image_builder.py`.

Shape robustness
----------------
Generated images range from 1x9 (Cancer, packed layout) to roughly 10x120
(Gene, attention_map). Any architecture that downsamples unconditionally will
break on the small end, so every pooling operation here is guarded by an
explicit spatial-size check and every model ends in AdaptiveAvgPool2d(1) or a
flatten. Nothing assumes a minimum image size.

The four architectures
----------------------
tabnetcnn     The existing model, unchanged. Reference point.
deep_cnn      Three conv blocks with 3x3 kernels, BatchNorm and guarded 2x2
              max-pooling. This is the architecture Table 4.2 of the thesis
              currently describes, which the implemented TabNetCNN is not.
small_resnet  Residual stack at constant resolution. Substantially deeper and
              higher-capacity, with a different optimisation profile.
pixel_mlp     Flattens the image and applies a plain MLP. This one matters
              most: an MLP over flattened pixels is invariant to the spatial
              arrangement up to a relabelling of first-layer weights, so it
              cannot exploit adjacency at all. If a layout's advantage over
              another layout survives here, that advantage was never spatial.
"""

from typing import Dict, Type

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _can_pool(h: int, w: int) -> bool:
    """A 2x2 max-pool is only meaningful if both dimensions survive it."""
    return h >= 2 and w >= 2


def _check_hw(image_height, image_width, name):
    if image_height is None or image_width is None:
        raise ValueError(
            f"{name} requires explicit image_height and image_width; "
            f"got {image_height}x{image_width}. Read them from "
            f"X_train_img.npy.shape[2:] before constructing the model."
        )


# ---------------------------------------------------------------------------
# 1. Reference architecture (identical to cnn/cnn_model.py)
# ---------------------------------------------------------------------------

class TabNetCNN(nn.Module):
    """Two conv blocks, BatchNorm, no spatial downsampling, global average pool.

    Reproduced verbatim from cnn/cnn_model.py so that the registry is
    self-contained and E1 cannot silently diverge from the reference model.
    Branch selection is on total pixel count (H*W), as in the original.
    """

    def __init__(self, n_classes, input_channels=1, image_height=None,
                 image_width=None, dropout=0.3):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width

        total_pixels = (
            image_height * image_width if image_height and image_width else None
        )

        if total_pixels is not None and total_pixels <= 16:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 8, kernel_size=2, padding=1),
                nn.BatchNorm2d(8), nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, kernel_size=2, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc_input = 16
        elif total_pixels is not None and total_pixels <= 100:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc_input = 32
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc_input = 64

        self.fc = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(self.fc_input, n_classes))

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B,C,H,W], got {tuple(x.shape)}")
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# 2. Deeper CNN with guarded pooling (the Table 4.2 architecture)
# ---------------------------------------------------------------------------

class DeepCNN(nn.Module):
    """Three 3x3 conv blocks with BatchNorm and 2x2 max-pooling where the
    spatial size permits it, then global average pooling.

    Unlike TabNetCNN this model does build a genuine hierarchy on images that
    are large enough to be pooled, so it is the natural architecture for
    testing whether the layouts encode structure a deeper network can use.
    """

    def __init__(self, n_classes, input_channels=1, image_height=None,
                 image_width=None, dropout=0.3, channels=(32, 64, 128)):
        super().__init__()
        _check_hw(image_height, image_width, "DeepCNN")

        h, w = int(image_height), int(image_width)
        layers = []
        c_in = input_channels
        self.n_pools = 0

        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
            if _can_pool(h, w):
                layers.append(nn.MaxPool2d(2))
                h, w = h // 2, w // 2
                self.n_pools += 1
            c_in = c_out

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.conv = nn.Sequential(*layers)
        self.fc_input = c_in
        self.fc = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(self.fc_input, n_classes))

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# 3. Small residual network at constant resolution
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class SmallResNet(nn.Module):
    """Residual stack with no downsampling, so it is safe on 1xF images.

    Deeper and wider than the reference model, with skip connections and a
    much larger effective receptive field. If the ranking of layouts is
    preserved here, that ranking is not an artefact of the reference model's
    two-layer, 5x5-receptive-field limitation.
    """

    def __init__(self, n_classes, input_channels=1, image_height=None,
                 image_width=None, dropout=0.3, width=32, n_blocks=3):
        super().__init__()
        _check_hw(image_height, image_width, "SmallResNet")

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1),
            nn.BatchNorm2d(width), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(width) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(width, n_classes))

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        x = self.pool(self.blocks(self.stem(x)))
        return self.fc(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# 4. Permutation-blind MLP control
# ---------------------------------------------------------------------------

class PixelMLP(nn.Module):
    """MLP over the flattened image: cannot exploit spatial adjacency.

    This is the control that turns E1 into a test of the thesis's central
    hypothesis rather than only of portability. Any two layouts that place the
    same feature values in the same number of pixels are, for this model,
    equivalent up to a permutation of the input weights. Differences that
    persist here are differences in *which features are present and how they
    are scaled*, not in geometry.

    LayerNorm is used rather than BatchNorm1d deliberately: BatchNorm1d raises
    on a training batch of size 1, which the smaller benchmarks (Glass, Horse)
    can produce as a final batch. LayerNorm removes that failure mode without
    changing the permutation argument.
    """

    def __init__(self, n_classes, input_channels=1, image_height=None,
                 image_width=None, dropout=0.3, hidden=(128, 64)):
        super().__init__()
        _check_hw(image_height, image_width, "PixelMLP")

        d_in = int(input_channels) * int(image_height) * int(image_width)
        layers = []
        prev = d_in
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(1)
        return self.net(x.reshape(x.size(0), -1))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCHITECTURES: Dict[str, Type[nn.Module]] = {
    "tabnetcnn": TabNetCNN,
    "deep_cnn": DeepCNN,
    "small_resnet": SmallResNet,
    "pixel_mlp": PixelMLP,
}


def available_architectures():
    return sorted(ARCHITECTURES.keys())


def build_model(arch_name, n_classes, input_channels=1, image_height=None,
                image_width=None, dropout=0.3, **extra):
    """Instantiate an architecture by name.

    `extra` is forwarded to the architecture constructor, so architecture-
    specific options (channels, width, n_blocks, hidden) can be tuned without
    changing this signature.
    """
    key = str(arch_name).strip().lower()
    if key not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture '{arch_name}'. "
            f"Available: {available_architectures()}"
        )
    return ARCHITECTURES[key](
        n_classes=n_classes,
        input_channels=input_channels,
        image_height=image_height,
        image_width=image_width,
        dropout=dropout,
        **extra,
    )


def count_parameters(model):
    """Trainable parameter count, for the capacity column of the E1 table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test across the full range of image shapes the pipeline produces.
    shapes = [
        (1, 9, "Cancer / packed"),
        (9, 1, "Cancer / packed_T"),
        (3, 3, "Cancer / naive"),
        (6, 10, "Cancer / step_sparse"),
        (6, 9, "Cancer / attention_map"),
        (8, 16, "Gene / packed"),
        (10, 120, "Gene / attention_map"),
    ]
    for h, w, label in shapes:
        for arch in available_architectures():
            m = build_model(arch, n_classes=3, input_channels=1,
                            image_height=h, image_width=w, dropout=0.3)
            x = torch.randn(4, 1, h, w)
            out = m(x)
            assert out.shape == (4, 3), (arch, h, w, out.shape)
            print(f"{label:28s} {h:>3}x{w:<3} {arch:14s} "
                  f"out={tuple(out.shape)} params={count_parameters(m):,}")
    print("\nAll architectures accept every shape the pipeline produces.")