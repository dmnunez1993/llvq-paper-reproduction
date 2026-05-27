# LLVQ Prototype Progress

This file tracks the exploratory LLVQ implementations added during the session.
The original `llvq.py` was restored and left as the simple reference script.

## Big Picture

The paper's LLVQ method is codebook-free:

- It does not store all lattice vectors.
- It indexes vectors hierarchically as shell -> class -> local symmetries.
- It uses structured Adoul-Barth/Golay search rather than brute-force scanning.
- For the best 2 bits/dim result in Table 7, it uses shape-gain:
  `norm(Lambda_24(12)) + 1 chi-gain bit`.

The current code is a set of prototypes. Some parts match the paper's spirit, but
the implementation is not yet mathematically equivalent to the paper.

## Files Added

### `llvq_gpu.py`

Torch/GPU version of the original finite codebook idea.

- Builds a finite shell database with Torch tensors.
- Stores `all_vectors` explicitly.
- Supports batched `quantize` and `dequantize`.
- Supports `mode="euclidean"` and `mode="angular"`.
- Uses GPU for filtering/scoring, but still materializes the codebook.

Status: useful baseline, not paper-faithful because it stores vectors.

### `llvq_gpu_shell.py`

GPU-expanded shell builder specialized for `max_coord <= 2`.

- Avoids some Python recursion from `llvq_gpu.py`.
- Still stores `all_vectors`.
- Adds CLI arguments for `M`, `max_coord`, `batch_vectors`, etc.

Status: faster explicit-codebook toy path, not paper-faithful.

### `llvq_implicit.py`

First codebook-free prototype.

- Does not store `all_vectors`.
- Stores compact segment metadata.
- Dequantizes by unranking an index into:
  shell/codeword/magnitude placements/sign pattern.
- Quantization streams candidates in chunks instead of storing the full codebook.
- Supports larger `max_coord` values by encoding odd/even magnitude patterns.
- Adds progress logging, timing, and size reporting.
- Adds on-disk cache in `.llvq_cache/`.

Status: codebook-free storage and dequantization prototype, but quantization is
still brute-force streaming over implicit candidates.

Useful commands:

```bash
uv run python -u llvq_implicit.py --M 5 --max_coord 2 --batch_vectors 2
uv run python -u llvq_implicit.py --M 8 --max_coord 4 --batch_vectors 1 --progress_every 1000000
```

### `llvq_triton_search.py`

Experimental Triton kernel for scoring implicit candidate chunks.

- Still generates candidate chunks via `llvq_implicit.py`.
- Replaces per-chunk `torch.cdist`/matmul scoring with a Triton min-reduction
  kernel.
- Verified against the Torch implicit search for Euclidean and angular modes.

Status: scoring kernel works, but the overall search is still brute-force over
candidate chunks. This does not solve huge search spaces.

### `llvq_adoul_barth.py`

Adoul-Barth-inspired structured search.

- Searches over structural segments instead of every candidate vector.
- For each segment, greedily places magnitudes on high `abs(x_i)` coordinates.
- Solves signs with a small dynamic program under the mod-4 sum constraint.
- Scores one strong candidate per segment.

Status: much faster than brute force, approximate, not the full paper's class
leader implementation.

Useful command:

```bash
uv run python -u llvq_adoul_barth.py --M 8 --max_coord 4 --batch_vectors 2
```

### `llvq_adoul_barth_fast_segments.py`

Faster segment metadata builder.

- Prunes impossible magnitudes by the norm budget `2M`.
- Precomputes valid magnitude-count patterns once per Golay codeword weight.
- Reuses those patterns across codewords with the same weight.
- Uses a separate cache:
  `.llvq_cache/fast_segments_v1_M{M}_C{max_coord}.pt`.

This fixed the apparent hang for large values like `M=20, max_coord=15`.

Example observed result:

```text
M=20, max_coord=15
build time: about 10.5s
segments: 54,483
represented candidates: 2,600,135,286,576
```

Useful command:

```bash
uv run python -u llvq_adoul_barth_fast_segments.py \
  --M 20 \
  --max_coord 15 \
  --batch_vectors 1 \
  --build_progress_every 512 \
  --progress_every 10000
```

### `llvq_adoul_barth_gpu_search.py`

GPU-batched scoring for the Adoul-Barth-style candidates.

- Uses the fast segment builder.
- Builds segment candidates in chunks.
- Scores the chunk on GPU as one tensor.
- Keeps the same result as `llvq_adoul_barth.py` on deterministic tests.

Status: speeds up scoring/reduction but candidate construction is still Python.

Useful command:

```bash
uv run python -u llvq_adoul_barth_gpu_search.py \
  --M 20 \
  --max_coord 15 \
  --batch_vectors 2 \
  --segment_chunk_size 262144
```

### `llvq_adoul_barth_triton_candidates.py`

Experimental GPU-vectorized segment selector.

- Originally attempted a Triton candidate-for-segment kernel, but that kernel
  had excessive JIT compile time.
- Replaced with a Torch CUDA-vectorized selector.
- Selects the best segment on GPU and then uses the exact Python
  `_candidate_for_segment` once for the selected segment.
- Verified on a deterministic test against `llvq_adoul_barth.py`.

Status: practical experimental GPU selector. Despite the filename, the active
selector is Torch CUDA, not the stalled Triton kernel.

Useful command:

```bash
uv run python -u llvq_adoul_barth_triton_candidates.py \
  --M 12 \
  --max_coord 5 \
  --batch_vectors 2
```

### `llvq_shape_gain.py`

Shape-gain prototype.

- Uses angular shape search.
- Normalizes the selected lattice shape.
- Adds a scalar chi-style gain quantizer.
- Reconstructs:

```python
recon = quantized_gain * normalized_shape
```

- Prints:
  - raw shape MSE
  - optimal unquantized gain MSE
  - quantized shape-gain MSE
  - total bits/vector and bits/dim

This addresses the issue where raw angular `recon` looked too small compared
with `x`.

Useful paper-like command:

```bash
uv run python -u llvq_shape_gain.py \
  --M 12 \
  --max_coord 5 \
  --gain_bits 1 \
  --batch_vectors 2
```

Observed example:

```text
raw shape mse:                3.6372
optimal unquantized gain mse: 1.4358
shape-gain mse:               1.4427
```

## Important Interpretations

### `batch_vectors`

Number of random 24-D vectors generated and quantized in one run.

```text
batch_vectors=2 -> x.shape == (2, 24)
```

It does not change the codebook/search space.

### `M`

Shell/radius parameter. Larger `M` means more shells and more candidates.

In this code, candidates satisfy roughly:

```text
sum(v_i^2) <= 2M
```

The paper uses `M`/shells, but not our `max_coord`.

### `max_coord`

Implementation-only coordinate bound. It is not a paper hyperparameter.

For a given `M`, values above `floor(sqrt(2M))` are usually pruned anyway.

Examples:

```text
M=12 -> max_coord around 5 is enough
M=13 -> max_coord around 6 is enough
M=20 -> max_coord around 7 is enough
M=30 -> max_coord around 8 is enough
```

### Angular Score

For `mode="angular"`:

```text
score = 1 - cosine_similarity
```

So `score=0.0866` means cosine similarity about `0.9134`.

### Bits Per Vector

Each index represents one 24-D block.

```text
bits/dim = bits/vector / 24
```

Example:

```text
49 bits/vector -> 49 / 24 = 2.04 bits/dim
```

## Paper-Likeness

Closest current command for the paper's Table 7 shape-gain idea:

```bash
uv run python -u llvq_shape_gain.py \
  --M 12 \
  --max_coord 5 \
  --gain_bits 1 \
  --batch_vectors 2
```

But this is still not equivalent to the paper.

## What Is Still Different From The Paper

1. True Leech shell counts are not matched.

   The paper's `Lambda_24(12)` shape code is around 47 shape bits/vector.
   Our `M=12` prototype reports around 33 shape bits/vector. This means the
   candidate set is not the same object.

2. `max_coord` does not exist in the paper.

   It is only a practical parameter in our simplified construction.

3. Search is approximate.

   The paper uses a structured Adoul-Barth/class-leader search. Our search uses
   greedy magnitude placement and sign fixing.

4. Gain quantization is approximate.

   `llvq_shape_gain.py` uses a sampled Lloyd quantizer over `sqrt(chi2_24)`.
   This is similar in spirit but not a full reproduction of the paper's gain
   setup and optimal scale handling.

5. Full hierarchical class indexing is missing.

   The paper indexes shell -> class -> local symmetries. Our implementation uses
   simplified segment metadata.

## Performance Notes

- Brute-force implicit scanning is too slow for huge candidate counts.
- Fast segment building helps a lot.
- Current Adoul-Barth-style search is still prototype-level speed.
- Quantize time around `0.2s` for one vector is too slow for real model
  quantization.
- Real speed would require:
  - true paper class leaders,
  - batched layer-level processing,
  - a fused GPU/Triton/CUDA kernel,
  - no Python loop over individual 24-D blocks.

## Recommended Next Steps

1. Implement true Leech shell/class leader tables.
2. Match the paper's shell cardinalities for `Lambda_24(M)`.
3. Replace simplified segment indexing with shell -> class -> symmetry indexing.
4. Improve shape-gain:
   - calibrated gain scale,
   - chi-gain quantizer tables,
   - optional optimal scales.
5. Batch quantization over many 24-D blocks from actual tensors.
6. Only then optimize kernels for real throughput.
