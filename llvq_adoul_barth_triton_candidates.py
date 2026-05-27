from __future__ import annotations

from math import ceil, log2
from time import perf_counter

import torch
import triton
import triton.language as tl

from llvq_adoul_barth import _candidate_for_segment, _score
from llvq_adoul_barth_fast_segments import FastSegmentShellDatabase
from llvq_implicit import DIM


@triton.jit
def _segment_select_kernel(
    x_ptr,
    odd_mask_ptr,
    even_mask_ptr,
    odd_counts_ptr,
    even_counts_ptr,
    odd_mags_ptr,
    even_mags_ptr,
    out_score_ptr,
    out_segment_ptr,
    n_segments,
    mode: tl.constexpr,
    dim: tl.constexpr,
    max_odd_mags: tl.constexpr,
    max_even_mags: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_segment = offsets < n_segments

    odd_mask = tl.load(odd_mask_ptr + offsets, mask=valid_segment, other=0).to(tl.int64)
    even_mask = tl.load(even_mask_ptr + offsets, mask=valid_segment, other=0).to(tl.int64)
    selected = tl.zeros((BLOCK_N,), dtype=tl.int64)
    dot = tl.zeros((BLOCK_N,), dtype=tl.float32)
    cand_norm2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_norm2 = tl.full((), 0.0, dtype=tl.float32)
    residue = tl.zeros((BLOCK_N,), dtype=tl.int32)
    best_flip_loss = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)

    for d in tl.static_range(0, dim):
        x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
        x_norm2 += x_val * x_val

    for mag_i_rev in tl.static_range(0, max_odd_mags):
        mag_i = max_odd_mags - 1 - mag_i_rev
        mag = tl.load(odd_mags_ptr + mag_i).to(tl.int32)
        count = tl.load(odd_counts_ptr + offsets * max_odd_mags + mag_i, mask=valid_segment, other=0)

        for pick in tl.static_range(0, dim):
            active = valid_segment & (pick < count)
            best_abs = tl.full((BLOCK_N,), -1.0, dtype=tl.float32)
            best_sign = tl.full((BLOCK_N,), 1, dtype=tl.int32)
            best_bit = tl.zeros((BLOCK_N,), dtype=tl.int64)

            for d in tl.static_range(0, dim):
                bit = (tl.full((BLOCK_N,), 1, dtype=tl.int64) << d)
                allowed = ((odd_mask & bit) != 0) & ((selected & bit) == 0) & active
                x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
                abs_val = tl.abs(x_val)
                better = allowed & (abs_val > best_abs)
                best_abs = tl.where(better, abs_val, best_abs)
                best_sign = tl.where(better, tl.where(x_val >= 0.0, 1, -1), best_sign)
                best_bit = tl.where(better, bit, best_bit)

            picked = active & (best_bit != 0)
            selected = tl.where(picked, selected | best_bit, selected)
            mag_f = mag.to(tl.float32)
            dot += tl.where(picked, mag_f * best_abs, 0.0)
            cand_norm2 += tl.where(picked, mag_f * mag_f, 0.0)
            signed_mag = best_sign * mag
            residue = tl.where(picked, (residue + signed_mag) % 4, residue)

            flip_loss = 2.0 * mag_f * best_abs
            delta = (-2 * signed_mag) % 4
            fixes = picked & (((residue + delta) % 4) == 0)
            best_flip_loss = tl.where(fixes & (flip_loss < best_flip_loss), flip_loss, best_flip_loss)

    for mag_i_rev in tl.static_range(0, max_even_mags):
        mag_i = max_even_mags - 1 - mag_i_rev
        mag = tl.load(even_mags_ptr + mag_i).to(tl.int32)
        count = tl.load(even_counts_ptr + offsets * max_even_mags + mag_i, mask=valid_segment, other=0)

        for pick in tl.static_range(0, dim):
            active = valid_segment & (pick < count)
            best_abs = tl.full((BLOCK_N,), -1.0, dtype=tl.float32)
            best_sign = tl.full((BLOCK_N,), 1, dtype=tl.int32)
            best_bit = tl.zeros((BLOCK_N,), dtype=tl.int64)

            for d in tl.static_range(0, dim):
                bit = (tl.full((BLOCK_N,), 1, dtype=tl.int64) << d)
                allowed = ((even_mask & bit) != 0) & ((selected & bit) == 0) & active
                x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
                abs_val = tl.abs(x_val)
                better = allowed & (abs_val > best_abs)
                best_abs = tl.where(better, abs_val, best_abs)
                best_sign = tl.where(better, tl.where(x_val >= 0.0, 1, -1), best_sign)
                best_bit = tl.where(better, bit, best_bit)

            picked = active & (best_bit != 0)
            selected = tl.where(picked, selected | best_bit, selected)
            mag_f = mag.to(tl.float32)
            dot += tl.where(picked, mag_f * best_abs, 0.0)
            cand_norm2 += tl.where(picked, mag_f * mag_f, 0.0)
            signed_mag = best_sign * mag
            residue = tl.where(picked, (residue + signed_mag) % 4, residue)

            # Even magnitudes usually cannot fix residue, but keep this general.
            flip_loss = 2.0 * mag_f * best_abs
            delta = (-2 * signed_mag) % 4
            fixes = picked & (((residue + delta) % 4) == 0)
            best_flip_loss = tl.where(fixes & (flip_loss < best_flip_loss), flip_loss, best_flip_loss)

    needs_fix = residue != 0
    can_fix = best_flip_loss < float("inf")
    dot = tl.where(needs_fix & can_fix, dot - best_flip_loss, dot)
    candidate_valid = valid_segment & ((residue == 0) | can_fix) & (cand_norm2 > 0.0)

    if mode == 0:
        scores = x_norm2 + cand_norm2 - 2.0 * dot
    else:
        scores = 1.0 - dot / tl.maximum(tl.sqrt(x_norm2 * cand_norm2), 1.0e-12)

    scores = tl.where(candidate_valid, scores, float("inf"))
    best_score = tl.min(scores, axis=0)
    best_pos = tl.argmin(scores, axis=0)
    out_pos = batch_id * tl.num_programs(1) + block_id
    tl.store(out_score_ptr + out_pos, best_score)
    tl.store(out_segment_ptr + out_pos, block_id * BLOCK_N + best_pos)


class LLVQAdoulBarthTritonCandidates:
    """
    Experimental Triton candidate-for-segment selector.

    Triton scans all structural segments in parallel using a greedy AB-style
    candidate approximation. The winning segment is then decoded with the
    existing exact Python `_candidate_for_segment`, so the returned index/vector
    are valid hierarchical LLVQ outputs.
    """

    def __init__(self, shell_db: FastSegmentShellDatabase, block_n: int = 128):
        if shell_db.total_count == 0:
            raise ValueError("shell_db is empty; call build(...) first")
        if shell_db.device.type != "cuda":
            raise ValueError("this experimental Triton path requires CUDA")
        self.db = shell_db
        self.block_n = triton.next_power_of_2(block_n)
        self._metadata = self._build_metadata()

    def _build_metadata(self) -> dict[str, torch.Tensor | int]:
        max_odd_mags = max(len(segment.odd_magnitudes) for segment in self.db.segments)
        max_even_mags = max(len(segment.even_magnitudes) for segment in self.db.segments)
        n = len(self.db.segments)

        odd_allowed = torch.zeros((n, DIM), dtype=torch.bool)
        even_allowed = torch.zeros((n, DIM), dtype=torch.bool)
        odd_masks = torch.zeros(n, dtype=torch.int64)
        even_masks = torch.zeros(n, dtype=torch.int64)
        odd_counts = torch.zeros((n, max_odd_mags), dtype=torch.int32)
        even_counts = torch.zeros((n, max_even_mags), dtype=torch.int32)

        odd_mags = torch.zeros(max_odd_mags, dtype=torch.int32)
        even_mags = torch.zeros(max_even_mags, dtype=torch.int32)
        if self.db.segments:
            odd_mags[: len(self.db.segments[0].odd_magnitudes)] = torch.tensor(
                self.db.segments[0].odd_magnitudes, dtype=torch.int32
            )
            even_mags[: len(self.db.segments[0].even_magnitudes)] = torch.tensor(
                self.db.segments[0].even_magnitudes, dtype=torch.int32
            )

        for i, segment in enumerate(self.db.segments):
            for pos in segment.odd_positions:
                odd_masks[i] |= 1 << pos
                odd_allowed[i, pos] = True
            for pos in segment.even_positions:
                even_masks[i] |= 1 << pos
                even_allowed[i, pos] = True
            odd_counts[i, : len(segment.odd_counts)] = torch.tensor(segment.odd_counts, dtype=torch.int32)
            even_counts[i, : len(segment.even_counts)] = torch.tensor(segment.even_counts, dtype=torch.int32)

        return {
            "odd_allowed": odd_allowed.to(self.db.device),
            "even_allowed": even_allowed.to(self.db.device),
            "odd_masks": odd_masks.to(self.db.device),
            "even_masks": even_masks.to(self.db.device),
            "odd_counts": odd_counts.to(self.db.device),
            "even_counts": even_counts.to(self.db.device),
            "odd_mags": odd_mags.to(self.db.device),
            "even_mags": even_mags.to(self.db.device),
            "max_odd_mags": max_odd_mags,
            "max_even_mags": max_even_mags,
        }

    def quantize(self, x: torch.Tensor, mode: str = "angular", return_vectors: bool = True):
        if mode not in {"euclidean", "angular"}:
            raise ValueError(f"unknown quantization mode: {mode}")

        x = torch.as_tensor(x, dtype=self.db.dtype, device=self.db.device).contiguous()
        single = x.ndim == 1
        x = x.reshape(-1, DIM).contiguous()
        segment_indices = self._select_segments_torch(x, mode).to("cpu").tolist()

        final_indices = []
        final_vectors = []
        final_scores = []
        for x_vec, segment_idx in zip(x, segment_indices, strict=True):
            candidate, index = _candidate_for_segment(x_vec.detach().to("cpu", dtype=torch.float32), self.db.segments[segment_idx])
            score = _score(
                x_vec.detach().to("cpu", dtype=torch.float32),
                candidate,
                mode=mode,
                x_norm=float(torch.linalg.norm(x_vec).item()),
            )
            final_indices.append(index)
            final_vectors.append(candidate)
            final_scores.append(score)

        idx = torch.tensor(final_indices, dtype=torch.long, device=self.db.device)
        vectors = torch.stack(final_vectors).to(device=self.db.device, dtype=self.db.dtype) if return_vectors else None
        scores = torch.tensor(final_scores, dtype=self.db.dtype, device=self.db.device)

        if single:
            idx = idx[0]
            scores = scores[0]
            vectors = vectors[0] if vectors is not None else None
        return idx, vectors, scores

    def dequantize(self, idx: int | torch.Tensor) -> torch.Tensor:
        return self.db.dequantize(idx)

    def _select_segments_torch(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """GPU-vectorized approximate segment selector.

        This replaces the too-heavy Triton JIT path. It evaluates one greedy
        candidate per segment in parallel with Torch CUDA ops, then returns the
        best segment per input vector. The final index/vector are still rebuilt
        exactly for the selected segment by `_candidate_for_segment`.
        """

        winners = []
        odd_allowed = self._metadata["odd_allowed"]
        even_allowed = self._metadata["even_allowed"]
        odd_counts = self._metadata["odd_counts"].to(torch.long)
        even_counts = self._metadata["even_counts"].to(torch.long)
        odd_mags = self._metadata["odd_mags"].to(self.db.dtype)
        even_mags = self._metadata["even_mags"].to(self.db.dtype)
        n_segments = odd_allowed.shape[0]
        rows = torch.arange(n_segments, device=self.db.device)

        for x_vec in x:
            abs_x = x_vec.abs()
            candidate = torch.zeros((n_segments, DIM), dtype=self.db.dtype, device=self.db.device)
            selected = torch.zeros((n_segments, DIM), dtype=torch.bool, device=self.db.device)

            for mag_i in range(int(self._metadata["max_odd_mags"]) - 1, -1, -1):
                counts = odd_counts[:, mag_i]
                max_count = int(counts.max().item())
                magnitude = odd_mags[mag_i]
                if max_count == 0 or float(magnitude.item()) == 0.0:
                    continue
                allowed = odd_allowed & ~selected
                values = abs_x[None, :].expand(n_segments, DIM).masked_fill(~allowed, -torch.inf)
                top_idx = values.topk(max_count, dim=1).indices
                take = torch.arange(max_count, device=self.db.device)[None, :] < counts[:, None]
                row_idx = rows[:, None].expand_as(top_idx)
                picked_rows = row_idx[take]
                picked_cols = top_idx[take]
                signs = torch.where(x_vec[picked_cols] >= 0, 1.0, -1.0).to(self.db.dtype)
                candidate[picked_rows, picked_cols] = signs * magnitude
                selected[picked_rows, picked_cols] = True

            for mag_i in range(int(self._metadata["max_even_mags"]) - 1, -1, -1):
                counts = even_counts[:, mag_i]
                max_count = int(counts.max().item())
                magnitude = even_mags[mag_i]
                if max_count == 0 or float(magnitude.item()) == 0.0:
                    continue
                allowed = even_allowed & ~selected
                values = abs_x[None, :].expand(n_segments, DIM).masked_fill(~allowed, -torch.inf)
                top_idx = values.topk(max_count, dim=1).indices
                take = torch.arange(max_count, device=self.db.device)[None, :] < counts[:, None]
                row_idx = rows[:, None].expand_as(top_idx)
                picked_rows = row_idx[take]
                picked_cols = top_idx[take]
                signs = torch.where(x_vec[picked_cols] >= 0, 1.0, -1.0).to(self.db.dtype)
                candidate[picked_rows, picked_cols] = signs * magnitude
                selected[picked_rows, picked_cols] = True

            residue = candidate.sum(dim=1).to(torch.long).remainder(4)
            needs_fix = residue != 0
            if needs_fix.any():
                values = candidate.to(torch.long)
                deltas = (-2 * values).remainder(4)
                fixes = ((residue[:, None] + deltas).remainder(4) == 0) & (values != 0)
                flip_loss = (2.0 * (candidate * x_vec[None, :]).abs()).masked_fill(~fixes, torch.inf)
                best_loss, best_col = flip_loss.min(dim=1)
                fixable = needs_fix & torch.isfinite(best_loss)
                if fixable.any():
                    candidate[rows[fixable], best_col[fixable]] *= -1

            valid = candidate.square().sum(dim=1) > 0
            if mode == "euclidean":
                scores = (candidate - x_vec[None, :]).square().sum(dim=1)
            else:
                denom = candidate.norm(dim=1).clamp_min(1.0e-12) * x_vec.norm().clamp_min(1.0e-12)
                scores = 1.0 - (candidate @ x_vec) / denom
            scores = scores.masked_fill(~valid, torch.inf)
            winners.append(scores.argmin())

        return torch.stack(winners).to(torch.long)


def demo(
    M: int = 20,
    max_coord: int = 15,
    batch_vectors: int = 2,
    input_scale: float = 3.0,
    mode: str = "angular",
    block_n: int = 128,
    device: str | None = "cuda",
    verbose: bool = True,
    build_progress_every: int = 512,
    cache_dir: str | None = ".llvq_cache",
    no_cache: bool = False,
) -> None:
    db = FastSegmentShellDatabase(device=device)
    build_start = perf_counter()
    db.build(
        M=M,
        max_coord=max_coord,
        verbose=verbose,
        progress_every=build_progress_every,
        cache_dir=cache_dir,
        use_cache=not no_cache,
    )
    build_seconds = perf_counter() - build_start

    print("device:", db.device)
    print("shell sizes:", db.shell_sizes())
    print("total implicit vectors:", db.total_count)
    print("structural segments:", len(db.segments))
    print(f"build time: {build_seconds:.6f}s")

    quantizer = LLVQAdoulBarthTritonCandidates(db, block_n=block_n)
    x = torch.randn(batch_vectors, DIM, device=db.device) * input_scale

    torch.cuda.synchronize()
    quantize_start = perf_counter()
    idx, recon, scores = quantizer.quantize(x, mode=mode)
    torch.cuda.synchronize()
    quantize_seconds = perf_counter() - quantize_start

    torch.cuda.synchronize()
    dequantize_start = perf_counter()
    timed_recon = quantizer.dequantize(idx)
    torch.cuda.synchronize()
    dequantize_seconds = perf_counter() - dequantize_start
    scale = (x * recon).sum(dim=1, keepdim=True) / recon.square().sum(dim=1, keepdim=True).clamp_min(1e-12)
    scaled_recon = scale * recon

    diff = x - scaled_recon
    index_bits = ceil(log2(db.total_count)) if db.total_count > 1 else 1
    packed_index_bytes = ceil(index_bits * idx.numel() / 8)

    print(f"quantize time: {quantize_seconds:.6f}s")
    print(f"dequantize time: {dequantize_seconds:.6f}s")
    print("timed dequantize matches:", torch.equal(recon, timed_recon))
    print(f"packed index size: {index_bits} bits/vector, {packed_index_bytes} bytes")
    print("indices:", idx)
    print("scores:", scores)
    print("x:", x)
    print("recon:", recon)
    print("x - recon:", diff)
    print("per-vector l2 error:", diff.norm(dim=1))
    print("mean squared error:", diff.square().mean())


def _argparse_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Triton candidate-for-segment LLVQ demo")
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--max_coord", type=int, default=15)
    parser.add_argument("--batch_vectors", type=int, default=2)
    parser.add_argument("--input_scale", type=float, default=3.0)
    parser.add_argument("--mode", choices=["euclidean", "angular"], default="angular")
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--build_progress_every", type=int, default=512)
    parser.add_argument("--cache_dir", type=str, default=".llvq_cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    demo(
        M=args.M,
        max_coord=args.max_coord,
        batch_vectors=args.batch_vectors,
        input_scale=args.input_scale,
        mode=args.mode,
        block_n=args.block_n,
        device=args.device,
        verbose=not args.quiet,
        build_progress_every=args.build_progress_every,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        _argparse_main()
    else:
        fire.Fire(demo)
