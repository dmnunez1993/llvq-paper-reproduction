from __future__ import annotations

from math import ceil, log2
from time import perf_counter

import torch

from llvq_adoul_barth import _candidate_for_segment
from llvq_adoul_barth_fast_segments import FastSegmentShellDatabase
from llvq_implicit import DIM


class LLVQAdoulBarthGPUSearch:
    """
    Batched GPU scorer for the Adoul-Barth-style segment search.

    Candidate construction is still CPU/Python because each segment has irregular
    placement/ranking metadata and depends on the current input vector. The
    expensive scoring/reduction is batched on GPU: each chunk builds many segment
    candidates, moves them to CUDA as one tensor, scores all candidates at once,
    and keeps the best.
    """

    def __init__(self, shell_db: FastSegmentShellDatabase, segment_chunk_size: int = 8192):
        if shell_db.total_count == 0:
            raise ValueError("shell_db is empty; call build(...) first")
        self.db = shell_db
        self.segment_chunk_size = segment_chunk_size

    def quantize(
        self,
        x: torch.Tensor,
        mode: str = "angular",
        return_vectors: bool = True,
        verbose: bool = False,
        progress_every: int = 10_000,
    ):
        if mode not in {"euclidean", "angular"}:
            raise ValueError(f"unknown quantization mode: {mode}")

        x = torch.as_tensor(x, dtype=self.db.dtype, device=self.db.device)
        single = x.ndim == 1
        x = x.reshape(-1, DIM)

        best_scores = torch.full((x.shape[0],), torch.inf, dtype=self.db.dtype, device=self.db.device)
        best_indices = torch.zeros((x.shape[0],), dtype=torch.long, device=self.db.device)
        best_vectors = torch.zeros((x.shape[0], DIM), dtype=self.db.dtype, device=self.db.device)

        if verbose:
            print(
                f"GPU-batched Adoul-Barth-style search over {len(self.db.segments)} segments "
                f"(segment_chunk_size={self.segment_chunk_size}, mode={mode})",
                flush=True,
            )

        for batch_idx, x_vec in enumerate(x):
            idx, vec, score = self._search_one(
                x_vec,
                mode=mode,
                verbose=verbose,
                progress_every=progress_every,
            )
            best_indices[batch_idx] = idx
            best_vectors[batch_idx] = vec
            best_scores[batch_idx] = score

            if verbose:
                print(
                    f"  vector {batch_idx + 1}/{x.shape[0]} | "
                    f"best_index={int(idx.item())} | best_score={float(score.item()):.6g}",
                    flush=True,
                )

        vectors = best_vectors if return_vectors else None
        if single:
            best_indices = best_indices[0]
            best_scores = best_scores[0]
            vectors = vectors[0] if vectors is not None else None
        return best_indices, vectors, best_scores

    def dequantize(self, idx: int | torch.Tensor) -> torch.Tensor:
        return self.db.dequantize(idx)

    def _search_one(
        self,
        x_vec: torch.Tensor,
        mode: str,
        verbose: bool,
        progress_every: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_cpu = x_vec.detach().to("cpu", dtype=torch.float32)
        x_gpu = x_vec.to(device=self.db.device, dtype=self.db.dtype)
        x_norm = torch.linalg.norm(x_gpu).clamp_min(1.0e-12)

        best_score = torch.tensor(float("inf"), dtype=self.db.dtype, device=self.db.device)
        best_index = torch.tensor(0, dtype=torch.long, device=self.db.device)
        best_vector = torch.zeros(DIM, dtype=self.db.dtype, device=self.db.device)
        next_report = progress_every

        for start in range(0, len(self.db.segments), self.segment_chunk_size):
            end = min(start + self.segment_chunk_size, len(self.db.segments))
            chunk = self.db.segments[start:end]

            candidates_cpu = []
            indices = []
            for segment in chunk:
                candidate, index = _candidate_for_segment(x_cpu, segment)
                candidates_cpu.append(candidate)
                indices.append(index)

            candidates = torch.stack(candidates_cpu).to(device=self.db.device, dtype=self.db.dtype)
            index_tensor = torch.tensor(indices, dtype=torch.long, device=self.db.device)

            if mode == "euclidean":
                scores = (candidates - x_gpu[None, :]).square().sum(dim=1)
            else:
                candidate_norms = torch.linalg.norm(candidates, dim=1).clamp_min(1.0e-12)
                cosine = (candidates @ x_gpu) / (candidate_norms * x_norm)
                scores = 1.0 - cosine

            chunk_score, chunk_pos = scores.min(dim=0)
            if chunk_score < best_score:
                best_score = chunk_score
                best_index = index_tensor[chunk_pos]
                best_vector = candidates[chunk_pos]

            searched = end
            if verbose and (searched >= next_report or searched == len(self.db.segments)):
                pct = 100.0 * searched / len(self.db.segments)
                print(
                    f"    searched {searched:8d}/{len(self.db.segments)} segments "
                    f"({pct:6.2f}%) | best_score={float(best_score.item()):.6g}",
                    flush=True,
                )
                while next_report <= searched:
                    next_report += progress_every

        return best_index, best_vector, best_score


def demo(
    M: int = 20,
    max_coord: int = 15,
    batch_vectors: int = 2,
    input_scale: float = 3.0,
    mode: str = "angular",
    device: str | None = None,
    segment_chunk_size: int = 8192,
    verbose: bool = True,
    progress_every: int = 10_000,
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

    quantizer = LLVQAdoulBarthGPUSearch(db, segment_chunk_size=segment_chunk_size)
    x = torch.randn(batch_vectors, DIM, device=db.device) * input_scale

    if db.device.type == "cuda":
        torch.cuda.synchronize(db.device)
    quantize_start = perf_counter()
    idx, recon, scores = quantizer.quantize(
        x,
        mode=mode,
        verbose=verbose,
        progress_every=progress_every,
    )
    if db.device.type == "cuda":
        torch.cuda.synchronize(db.device)
    quantize_seconds = perf_counter() - quantize_start

    if db.device.type == "cuda":
        torch.cuda.synchronize(db.device)
    dequantize_start = perf_counter()
    timed_recon = quantizer.dequantize(idx)
    if db.device.type == "cuda":
        torch.cuda.synchronize(db.device)
    dequantize_seconds = perf_counter() - dequantize_start

    diff = x - recon
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

    parser = argparse.ArgumentParser(description="GPU-batched Adoul-Barth-style LLVQ search demo")
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--max_coord", type=int, default=15)
    parser.add_argument("--batch_vectors", type=int, default=2)
    parser.add_argument("--input_scale", type=float, default=3.0)
    parser.add_argument("--mode", choices=["euclidean", "angular"], default="angular")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--segment_chunk_size", type=int, default=8192)
    parser.add_argument("--progress_every", type=int, default=10_000)
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
        device=args.device,
        segment_chunk_size=args.segment_chunk_size,
        verbose=not args.quiet,
        progress_every=args.progress_every,
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
