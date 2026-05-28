
"""Calibration data loader — DCLM-edu sequences for Hessian estimation.

The paper computes (GPTQ-style) layer-wise input Hessians on 6 100 sequences
from DCLM-edu (Li et al. 2024, Allal et al. 2025), at a context length of
4 096 tokens. This matches the calibration set size used in Quip# (Tseng et
al. 2024a) for apples-to-apples comparison.

Paper reference: §5.1

TODO:
  - Load DCLM-edu via `datasets.load_dataset("HuggingFaceTB/dclm-edu", ...)`
    (verify the exact HF path matches what the paper used).
  - Sample 6 100 sequences, tokenize at the target model's context length.
  - Cache tokenized sequences to disk so calibration is one-shot.
  - Provide a `make_calibration_loader(tokenizer, n_seqs, ctx, batch_size)`.
"""

from __future__ import annotations

from pathlib import Path

import torch


def make_calibration_loader(
    tokenizer,                # transformers tokenizer
    n_seqs: int = 6100,
    ctx_len: int = 4096,
    batch_size: int = 1,
    cache_dir: Path | str = "calibration_cache",
) -> torch.utils.data.DataLoader:
    """Yield batches of tokenized DCLM-edu sequences for Hessian estimation."""
    raise NotImplementedError("see paper §5.1")
