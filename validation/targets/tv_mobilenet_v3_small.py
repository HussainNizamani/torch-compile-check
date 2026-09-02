"""torchvision's MobileNetV3-Small, random-initialized, as a validation target.

See ``validation/targets/tv_resnet18.py`` for the shared rationale (no
network access, ``weights=None``, reduced input size, ``eval()`` for batch
norm determinism). This architecture adds depthwise-separable convolutions,
squeeze-and-excitation blocks, and the hard-swish/hard-sigmoid activations
-- lowering paths ``cases/`` does not exercise and that a mobile-oriented
architecture is exactly where they matter.
"""

from __future__ import annotations

import torch
import torchvision

model = torchvision.models.mobilenet_v3_small(weights=None)
model.eval()

inputs = (torch.randn(1, 3, 64, 64),)
