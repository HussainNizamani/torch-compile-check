"""torchvision's EfficientNet-B0, random-initialized, as a validation target.

See ``validation/targets/tv_resnet18.py`` for the shared rationale. This
architecture adds MBConv blocks (inverted residuals with squeeze-and-
excitation) and stochastic depth; stochastic depth is why ``eval()`` matters
here even more than for the other two vision targets -- in train mode it
randomly drops whole blocks per call, which two calls of the same eager
model would already disagree on before compilation enters the picture.
"""

from __future__ import annotations

import torch
import torchvision

model = torchvision.models.efficientnet_b0(weights=None)
model.eval()

inputs = (torch.randn(1, 3, 64, 64),)
