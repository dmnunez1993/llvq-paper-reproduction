from __future__ import annotations

from pathlib import Path
from time import perf_counter

import torch

from llvq_adoul_barth import LLVQAdoulBarth
from llvq_implicit import (
    DIM,
    GolayCode,
    ImplicitShellDatabase,
    Segment,
    _bounded_counts,
    _count_compositions,
    _expanded_magnitudes,
    _group_assignment_count,
    _valid_sign_count,
)


class FastSegmentShellDatabase(ImplicitShellDatabase):
    """
    Faster implicit segment builder for large `max_coord`.

    This is not a fully GPU-only builder: the final output is an irregular Python
    list of Segment records because `LLVQAdoulBarth` consumes that structure.
    The speedup comes from:

    - pruning impossible magnitudes by the norm budget `2M`;
    - precomputing valid magnitude-count patterns once per Golay codeword weight;
    - reusing those patterns across all codewords with the same weight;
    - using Torch/GPU for Golay codeword generation and weight extraction.
    """

    def build(
        self,
        M: int,
        max_coord: int = 15,
        verbose: bool = False,
        progress_every: int = 128,
        cache_dir: str | Path | None = ".llvq_cache",
        use_cache: bool = True,
    ) -> None:
        if M < 2:
            raise ValueError("M must be at least 2")
        if max_coord < 1:
            raise ValueError("max_coord must be at least 1")

        cache_path = _fast_cache_path(cache_dir, M, max_coord)
        if use_cache and cache_path is not None and cache_path.exists():
            if verbose:
                print(f"loading fast segment cache: {cache_path}", flush=True)
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.M = payload["M"]
            self.max_coord = payload["max_coord"]
            self.shell_offsets = payload["shell_offsets"]
            self.shell_counts = payload["shell_counts"]
            self.segments = payload["segments"]
            self.total_count = payload["total_count"]
            if verbose:
                print(
                    f"loaded cache: total_vectors={self.total_count}, "
                    f"segments={len(self.segments)}, shell_sizes={self.shell_counts}",
                    flush=True,
                )
            return

        self.M = M
        self.max_coord = max_coord
        self.shell_offsets.clear()
        self.shell_counts.clear()
        self.segments.clear()
        self.total_count = 0

        max_norm2 = 2 * M
        odd_magnitudes = tuple(m for m in range(1, max_coord + 1, 2) if m * m <= max_norm2)
        even_magnitudes = tuple(m for m in range(2, max_coord + 1, 2) if m * m <= max_norm2)

        if not odd_magnitudes:
            odd_magnitudes = ()
        if verbose:
            pruned_odd = tuple(range(1, max_coord + 1, 2))
            pruned_even = tuple(range(2, max_coord + 1, 2))
            print(
                f"fast implicit build on {self.device} (M={M}, max_coord={max_coord})",
                flush=True,
            )
            print(
                f"  norm budget={max_norm2}; odd magnitudes {pruned_odd} -> {odd_magnitudes}; "
                f"even magnitudes {pruned_even} -> {even_magnitudes}",
                flush=True,
            )

        self.golay = GolayCode(self.device)
        codewords = self.golay.codewords
        weights = codewords.sum(dim=1).to(torch.int64)
        codewords_cpu = codewords.cpu().tolist()
        weights_cpu = weights.cpu().tolist()
        unique_weights = sorted(set(weights_cpu))

        pattern_catalog = {
            weight: _patterns_for_weight(
                weight=weight,
                even_positions_count=DIM - weight,
                odd_magnitudes=odd_magnitudes,
                even_magnitudes=even_magnitudes,
                max_norm2=max_norm2,
            )
            for weight in unique_weights
        }

        if verbose:
            print(
                "  precomputed pattern specs by codeword weight: "
                + ", ".join(f"w={w}: {len(pattern_catalog[w])}" for w in unique_weights),
                flush=True,
            )

        pending: dict[int, list[tuple]] = {}
        progress_shell_counts: dict[int, int] = {}
        progress_segments = 0
        progress_candidates = 0

        for codeword_idx, (codeword, weight) in enumerate(zip(codewords_cpu, weights_cpu, strict=True), start=1):
            odd_positions = tuple(i for i, bit in enumerate(codeword) if bit)
            even_positions = tuple(i for i, bit in enumerate(codeword) if not bit)
            codeword_segments = 0
            codeword_candidates = 0

            for spec in pattern_catalog[weight]:
                (
                    shell,
                    odd_counts,
                    even_counts,
                    odd_place_count,
                    even_place_count,
                    sign_count,
                    count,
                ) = spec
                pending.setdefault(shell, []).append(
                    (
                        codeword_idx - 1,
                        odd_positions,
                        even_positions,
                        odd_magnitudes,
                        even_magnitudes,
                        odd_counts,
                        even_counts,
                        odd_place_count,
                        even_place_count,
                        sign_count,
                        count,
                    )
                )
                progress_shell_counts[shell] = progress_shell_counts.get(shell, 0) + count
                progress_segments += 1
                progress_candidates += count
                codeword_segments += 1
                codeword_candidates += count

            if verbose and (
                codeword_idx == 1
                or codeword_idx % progress_every == 0
                or codeword_idx == len(codewords_cpu)
            ):
                pct = 100.0 * codeword_idx / len(codewords_cpu)
                print(
                    f"  build codeword {codeword_idx:4d}/{len(codewords_cpu)} ({pct:6.2f}%) | "
                    f"added_segments={codeword_segments:7d} | "
                    f"added_candidates={codeword_candidates:14d} | "
                    f"segments={progress_segments:9d} | "
                    f"candidates={progress_candidates:14d}",
                    flush=True,
                )

        global_start = 0
        for shell in sorted(pending):
            self.shell_offsets[shell] = global_start
            shell_start = 0
            for item in pending[shell]:
                (
                    codeword_idx,
                    odd_positions,
                    even_positions,
                    odd_mags,
                    even_mags,
                    odd_counts,
                    even_counts,
                    odd_place_count,
                    even_place_count,
                    sign_count,
                    count,
                ) = item
                self.segments.append(
                    Segment(
                        shell=shell,
                        codeword_idx=codeword_idx,
                        odd_positions=odd_positions,
                        even_positions=even_positions,
                        odd_magnitudes=odd_mags,
                        even_magnitudes=even_mags,
                        odd_counts=odd_counts,
                        even_counts=even_counts,
                        odd_place_count=odd_place_count,
                        even_place_count=even_place_count,
                        sign_count=sign_count,
                        count=count,
                        shell_start=shell_start,
                        global_start=global_start + shell_start,
                    )
                )
                shell_start += count
            self.shell_counts[shell] = shell_start
            global_start += shell_start

        self.total_count = global_start

        if verbose:
            print(
                f"finished: total_vectors={self.total_count}, "
                f"segments={len(self.segments)}, shell_sizes={self.shell_counts}",
                flush=True,
            )

        if use_cache and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 1,
                    "builder": "fast_segments",
                    "M": self.M,
                    "max_coord": self.max_coord,
                    "shell_offsets": self.shell_offsets,
                    "shell_counts": self.shell_counts,
                    "segments": self.segments,
                    "total_count": self.total_count,
                },
                cache_path,
            )
            if verbose:
                print(f"saved fast segment cache: {cache_path}", flush=True)


def _patterns_for_weight(
    weight: int,
    even_positions_count: int,
    odd_magnitudes: tuple[int, ...],
    even_magnitudes: tuple[int, ...],
    max_norm2: int,
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...], int, int, int, int], ...]:
    out = []

    if weight > 0 and not odd_magnitudes:
        return ()

    for odd_counts in _count_compositions(weight, len(odd_magnitudes)):
        odd_norm2 = sum(c * (m * m) for c, m in zip(odd_counts, odd_magnitudes, strict=True))
        if odd_norm2 > max_norm2:
            continue

        even_budget = max_norm2 - odd_norm2
        for even_counts in _bounded_counts(even_positions_count, even_magnitudes, even_budget):
            norm2 = odd_norm2 + sum(c * (m * m) for c, m in zip(even_counts, even_magnitudes, strict=True))
            if norm2 < 4 or norm2 > max_norm2 or norm2 % 2:
                continue

            sign_magnitudes = _expanded_magnitudes(odd_magnitudes, odd_counts)
            sign_magnitudes += _expanded_magnitudes(even_magnitudes, even_counts)
            sign_count = _valid_sign_count(sign_magnitudes)
            if sign_count == 0:
                continue

            odd_place_count = _group_assignment_count(weight, odd_counts)
            even_place_count = _group_assignment_count(even_positions_count, even_counts)
            count = odd_place_count * even_place_count * sign_count
            out.append((norm2 // 2, odd_counts, even_counts, odd_place_count, even_place_count, sign_count, count))

    return tuple(out)


def _fast_cache_path(cache_dir: str | Path | None, M: int, max_coord: int) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"fast_segments_v1_M{M}_C{max_coord}.pt"


def demo(
    M: int = 20,
    max_coord: int = 15,
    batch_vectors: int = 2,
    input_scale: float = 3.0,
    mode: str = "angular",
    device: str | None = None,
    verbose: bool = True,
    progress_every: int = 10_000,
    build_progress_every: int = 128,
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

    quantizer = LLVQAdoulBarth(db)
    x = torch.randn(batch_vectors, DIM, device=db.device) * input_scale

    quantize_start = perf_counter()
    idx, recon, scores = quantizer.quantize(
        x,
        mode=mode,
        verbose=verbose,
        progress_every=progress_every,
    )
    quantize_seconds = perf_counter() - quantize_start

    dequantize_start = perf_counter()
    timed_recon = quantizer.dequantize(idx)
    dequantize_seconds = perf_counter() - dequantize_start

    diff = x - recon
    print(f"quantize time: {quantize_seconds:.6f}s")
    print(f"dequantize time: {dequantize_seconds:.6f}s")
    print("timed dequantize matches:", torch.equal(recon, timed_recon))
    print("indices:", idx)
    print("scores:", scores)
    print("x:", x)
    print("recon:", recon)
    print("x - recon:", diff)
    print("per-vector l2 error:", diff.norm(dim=1))
    print("mean squared error:", diff.square().mean())


def _argparse_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fast-segment Adoul-Barth-style LLVQ demo")
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--max_coord", type=int, default=15)
    parser.add_argument("--batch_vectors", type=int, default=2)
    parser.add_argument("--input_scale", type=float, default=3.0)
    parser.add_argument("--mode", choices=["euclidean", "angular"], default="angular")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--progress_every", type=int, default=10_000)
    parser.add_argument("--build_progress_every", type=int, default=128)
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
