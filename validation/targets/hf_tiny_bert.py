"""A tiny, random-initialized HF BERT, as a validation target.

PLAN.md "Real-world validation set" names a transformer as one of the
architecture families the fixture-sized corpus does not cover -- attention,
layer norm, and an embedding lookup as the first op rather than a
convolution. ``transformers`` is a validation-only extra (see
``docs/validation.md``), not a ``pyproject.toml`` dependency, so this file
is not always importable; ``validation/run.py`` checks for the package
before invoking ``compile-check`` on this target at all, and reports the
target as skipped rather than a tool error when it is absent, which is the
"skip cleanly" the slice brief asks for. Run directly with
``compile-check`` in an environment without ``transformers`` installed,
this file still fails at import (a plain ``ModuleNotFoundError``, caught by
discovery and reported as a ``DiscoveryError``, exit 2) rather than at some
later, more confusing point.

``BertConfig`` here is not a smaller version of a shipped checkpoint; every
weight is randomly initialized from this config, and no network access
happens at import or at construction. Vocabulary, hidden size, layer count,
and sequence length are all reduced from a real BERT-base config
(30522/768/12/512) to keep a CPU-only, 4-core compile fast:
``vocab_size=1000``, ``hidden_size=64``, ``num_hidden_layers=2``,
``num_attention_heads=2``, ``intermediate_size=128``,
``max_position_embeddings=32``.
"""

from __future__ import annotations

import torch
from transformers import BertConfig, BertModel

_config = BertConfig(
    vocab_size=1000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=128,
    max_position_embeddings=32,
)

model = BertModel(_config)
model.eval()


def get_inputs() -> dict[str, torch.Tensor]:
    """A fixed batch of token ids, seeded locally so re-importing is stable."""
    generator = torch.Generator().manual_seed(1234)
    input_ids = torch.randint(0, 1000, (1, 16), generator=generator)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
