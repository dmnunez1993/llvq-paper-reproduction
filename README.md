# LLVQ — Leech Lattice Vector Quantization (reproduction)

A reproduction attempt of **"Leech Lattice Vector Quantization for Efficient LLM Compression"**
by van der Ouderaa, van Baalen, Whatmough, and Nagel (Qualcomm AI Research, March 2026).

- Paper PDF: [`paper/llvq-paper-2603.11021v1.pdf`](paper/llvq-paper-2603.11021v1.pdf)
- arXiv: <https://arxiv.org/abs/2603.11021>

## Method, in one paragraph

LLVQ quantizes LLM weights by chopping each row of every linear-layer weight matrix
into 24-dimensional blocks and replacing each block with the nearest point of the
**Leech lattice Λ₂₄** — the densest known sphere packing in 24 dimensions. Crucially,
no codebook is stored: lattice points are addressed by integer indices through a
hierarchical scheme (shell → class → local symmetries) built on the extended binary
Golay code G₂₄. Quantization runs in two flavours: *spherical shaping* (ball-cut Λ₂₄)
and *shape–gain* (direction quantized on Λ₂₄ shells, magnitude via a scalar χ₂₄ code,
with optimal post-shape scale). GPTQ-style local-Hessian corrections — generalised
from scalars to 24-D blocks — bring the per-layer MSE down, and randomized Hadamard
rotations (Quip#/Quarot-style) decorrelate weights before quantization. The paper
reports state-of-the-art 2-bit PTQ on Llama-2, Llama-3, Ministral-3, and Qwen-v3.

## Repo status

This repository currently contains **layout + stubs only** — every module declares
its public API and points back to a paper section, but no algorithms are
implemented yet. Use the table below to track reproduction progress.

| Paper section / feature                                  | Module                              | Status |
| -------------------------------------------------------- | ----------------------------------- | :----: |
| §2.3 Extended Golay code G₂₄                             | `llvq/lattice/golay.py`             |   ☐    |
| §2.1–2.2 Λ₂₄ from G₂₄ (even/odd cosets)                  | `llvq/lattice/leech.py`             |   ☐    |
| §2.2 + Table 1: shell counts n(m), N(M)                  | `llvq/lattice/shells.py`            |   ☐    |
| §2.4–2.6: class leaders, subclass counts                 | `llvq/lattice/classes.py`           |   ☐    |
| §3 Adoul–Barth single-shell NN search                    | `llvq/search/adoul_barth.py`        |   ☐    |
| §3.1 Multi-shell extension                               | `llvq/search/multishell.py`         |   ☐    |
| §3.1 Angular / cosine scoring                            | `llvq/search/angular.py`            |   ☐    |
| §3.2 Hierarchical encoder (vector → index)               | `llvq/indexing/encode.py`           |   ☐    |
| §3.3 Dequantizer (index → vector)                        | `llvq/indexing/decode.py`           |   ☐    |
| §3.1 Shape (direction) quantizer                         | `llvq/shape_gain/shape.py`          |   ☐    |
| App. B Gain (magnitude) quantizer matched to √χ₂₄        | `llvq/shape_gain/gain.py`           |   ☐    |
| App. D.1 Optimal post-shape scales                       | `llvq/shape_gain/optimal_scales.py` |   ☐    |
| App. C Spherical-shaping variant (ball cut)              | `llvq/quantizer/spherical_shaping.py` |  ☐   |
| Main LLVQ shape–gain quantizer                           | `llvq/quantizer/shape_gain.py`      |   ☐    |
| App. D.2 Hessian-based GPTQ-style corrections            | `llvq/corrections/gptq.py`          |   ☐    |
| §5.3 Random Hadamard rotations                           | `llvq/rotations/hadamard.py`        |   ☐    |
| §3.5 Parallel dequantizer kernel (Python ref)            | `llvq/kernels/dequantizer.py`       |   ☐    |
| §5 Layer-wise PTQ pipeline (Llama-2 7B)                  | `llvq/llm/pipeline.py`              |   ☐    |
| §5.1 DCLM-edu calibration loader                         | `llvq/llm/calibration.py`           |   ☐    |
| §5.2 Optional scale fine-tuning                          | `llvq/llm/finetune.py`              |   ☐    |
| Fig. 1, Table 4: Gaussian-source SQNR-vs-rate            | `scripts/gaussian_benchmark.py`     |   ☐    |
| Table 3 / 5: Wikitext-2 perplexity, MMLU, CSR            | `scripts/eval_*.py`                 |   ☐    |

CUDA / Triton kernel and models other than Llama-2 7B are explicitly out of scope
for now.

## Quickstart (pixi)

This project uses [pixi](https://pixi.sh) for environment and dependency
management — declared inline in `pyproject.toml` under `[tool.pixi.*]`.

```bash
# Install the default environment (CUDA + dev tooling)
pixi install

# CPU-only contributors:
pixi install -e cpu

# Run the test scaffolding (everything is xfail until implementations land)
pixi run test

# Open an interactive shell with the env activated
pixi shell
```

Available pixi tasks (see `[tool.pixi.tasks]` in `pyproject.toml`):

| Task              | Command                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------- |
| `test`            | `pytest tests/`                                                                         |
| `lint` / `format` | `ruff check / format llvq tests scripts`                                                |
| `gaussian`        | `python scripts/gaussian_benchmark.py`  *(reproduces Fig. 1)*                           |
| `precompute`      | `python scripts/precompute_tables.py`  *(builds Golay codewords, shell tables, etc.)*   |
| `quantize-llama2` | `python scripts/quantize_llm.py --config configs/llama2_7b_llvq_sg_2bit.yaml`           |
| `ppl`             | `python scripts/eval_perplexity.py`                                                     |
| `downstream`      | `python scripts/eval_downstream.py`  *(requires the `eval` environment for lm-eval)*    |

## Repository layout

```
llvq/                # main package
  lattice/           # §2: Λ₂₄ from Golay code G₂₄, shells, classes
  search/            # §3: nearest-neighbour search (Adoul–Barth + multi-shell + angular)
  indexing/          # §3.2–3.3: bijective encoder / dequantizer
  shape_gain/        # §3.1, App. B/C/D.1: shape (direction) + gain (magnitude)
  quantizer/         # pluggable Quantizer ABC + spherical-shaping & shape-gain variants
  corrections/       # App. D.2: local-Hessian GPTQ-style corrections
  rotations/         # §5.3: Hadamard preprocessing (Quip# / Quarot style)
  kernels/           # §3.5: parallel dequantizer (Python ref; CUDA later)
  llm/               # end-to-end LLM PTQ pipeline (Llama-2 7B first)

configs/             # YAML experiment configs (one per row of Table 3)
scripts/             # CLI entry points (quantize, evaluate, gaussian benchmark)
tests/               # pytest unit tests for the math layer
notebooks/           # exploratory Jupyter notebooks (optional)
paper/               # paper PDF
```

## Current limitations

- Every algorithmic function raises `NotImplementedError`. Start from `tests/test_golay.py` and work outward.
- The dequantizer is intended as a vectorized PyTorch reference. Triton/CUDA acceleration is a separate milestone.
- Only Llama-2 7B is wired up in `llvq/llm/models/`. Llama-3 / Ministral-3 / Qwen-v3 require additional adapters.
- `lm-eval` integration in `scripts/eval_downstream.py` is signature-only.
