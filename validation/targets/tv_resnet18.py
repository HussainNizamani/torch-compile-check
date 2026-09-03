"""torchvision's ResNet-18, random-initialized, as a validation target.

PLAN.md "Real-world validation set": a public, CPU-runnable model that
exercises torch-compile-check on architecture shapes the fixture-sized regression
corpus (``cases/``) does not -- residual blocks, batch norm, adaptive
pooling. ``weights=None`` is deliberate: this suite runs offline and at
import time, per ``validation/run.py``'s no-network rule, so every target
here is randomly initialized rather than downloaded.

Input size reduced from the usual 224x224 to 64x64 (still well above the
architecture's minimum, since every downsampling stage still has spatial
extent left afterward) to keep a CPU-only, 4-core compile in the tens of
seconds rather than minutes; ``validation/run.py`` runs six targets across
three backends, twice each, and disk and wall time are shared with the rest
of the office.

``model.eval()`` matters here in a way it does not for ``cases/mlp.py``:
this architecture has batch norm, whose running-stats update in train mode
is a form of hidden state that would make two calls with identical inputs
disagree even under eager alone, which is noise this oracle exists to
filter out, not signal for it to report.
"""

from __future__ import annotations

import torch
import torchvision

model = torchvision.models.resnet18(weights=None)
model.eval()

inputs = (torch.randn(1, 3, 64, 64),)
