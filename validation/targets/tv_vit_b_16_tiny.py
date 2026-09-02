"""A reduced Vision Transformer, built from torchvision's own
``VisionTransformer`` class rather than the ``vit_b_16`` preset, as a
validation target.

See ``validation/targets/tv_resnet18.py`` for the shared rationale (no
network access, random init, ``eval()``). ``vit_b_16`` at its usual
224x224/patch-16/12-layer/768-hidden configuration compiles in minutes on a
CPU-only, 4-core box, which is not a cost this suite can pay six times over
across three backends; what a validation run needs from this architecture
is attention and layer-norm coverage, not the production-sized config. What
was reduced, concretely, against ``vit_b_16``'s 224/16/12/12/768/3072:

- image size 224 -> 32 (still divisible by the patch size, so the patch grid
  is exact: 4x4 = 16 patches plus the class token)
- patch size 16 -> 8
- transformer layers 12 -> 2
- attention heads 12 -> 2
- hidden dim 768 -> 32
- MLP dim 3072 -> 64

This is the same architecture -- patch embedding, class token, learned
positional embedding, pre-norm transformer encoder blocks, attention -- at a
size that compiles in seconds rather than minutes.
"""

from __future__ import annotations

import torch
import torchvision

model = torchvision.models.VisionTransformer(
    image_size=32,
    patch_size=8,
    num_layers=2,
    num_heads=2,
    hidden_dim=32,
    mlp_dim=64,
    num_classes=10,
)
model.eval()

inputs = (torch.randn(1, 3, 32, 32),)
