from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, log2
from time import perf_counter

import torch

from llvq_implicit import DIM, ImplicitShellDatabase, Segment, _valid_sign_patterns


@dataclass
class SearchResult:
    index: int
    vector: torch.Tensor
    score: float


class LLVQAdoulBarth:
    """
    Adoul-Barth-inspired structured LLVQ search.

    This is not the full paper implementation with complete Leech class tables.
    It uses the implicit segments from `llvq_implicit.py` and performs a fast
    leader-style search:

    1. For each segment/magnitude pattern, place larger magnitudes on larger
       |x_i| coordinates inside the allowed Golay parity positions.
    2. Solve the sign assignment exactly for the Conway-Sloane sum mod 4 rule
       with a tiny dynamic program over residues.
    3. Score one strong candidate per segment instead of brute-force scanning
       every placement/sign vector.

    This changes search from "all candidates" to "all structural segments", so it
    is much faster but approximate.
    """

    def __init__(self, shell_db: ImplicitShellDatabase):
        if shell_db.total_count == 0:
            raise ValueError("shell_db is empty; call build(...) first")
        self.db = shell_db

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

        best_indices = []
        best_vectors = []
        best_scores = []

        if verbose:
            print(
                f"Adoul-Barth-style search over {len(self.db.segments)} structural segments "
                f"representing {self.db.total_count} implicit candidates",
                flush=True,
            )

        for batch_idx, x_vec in enumerate(x):
            result = self._search_one(x_vec, mode, verbose=verbose, progress_every=progress_every)
            best_indices.append(result.index)
            best_vectors.append(result.vector)
            best_scores.append(result.score)

            if verbose:
                print(
                    f"  vector {batch_idx + 1}/{x.shape[0]} | "
                    f"best_index={result.index} | best_score={result.score:.6g}",
                    flush=True,
                )

        idx = torch.tensor(best_indices, dtype=torch.long, device=self.db.device)
        scores = torch.tensor(best_scores, dtype=self.db.dtype, device=self.db.device)
        vectors = torch.stack(best_vectors).to(device=self.db.device, dtype=self.db.dtype) if return_vectors else None

        if single:
            idx = idx[0]
            scores = scores[0]
            vectors = vectors[0] if vectors is not None else None

        return idx, vectors, scores

    def dequantize(self, idx: int | torch.Tensor) -> torch.Tensor:
        return self.db.dequantize(idx)

    def _search_one(
        self,
        x: torch.Tensor,
        mode: str,
        verbose: bool = False,
        progress_every: int = 10_000,
    ) -> SearchResult:
        x_cpu = x.detach().to("cpu", dtype=torch.float32)
        x_norm = float(torch.linalg.norm(x_cpu).item())
        best = SearchResult(index=0, vector=torch.zeros(DIM), score=float("inf"))

        for segment_idx, segment in enumerate(self.db.segments, start=1):
            candidate, index = _candidate_for_segment(x_cpu, segment)
            score = _score(x_cpu, candidate, mode=mode, x_norm=x_norm)

            if score < best.score:
                best = SearchResult(index=index, vector=candidate, score=score)

            if verbose and (segment_idx % progress_every == 0 or segment_idx == len(self.db.segments)):
                pct = 100.0 * segment_idx / len(self.db.segments)
                print(
                    f"    searched {segment_idx:8d}/{len(self.db.segments)} segments "
                    f"({pct:6.2f}%) | best_score={best.score:.6g}",
                    flush=True,
                )

        best.vector = best.vector.to(device=self.db.device, dtype=self.db.dtype)
        return best


def _candidate_for_segment(x: torch.Tensor, segment: Segment) -> tuple[torch.Tensor, int]:
    odd_groups = _greedy_magnitude_groups(
        x,
        segment.odd_positions,
        segment.odd_magnitudes,
        segment.odd_counts,
    )
    even_groups = _greedy_magnitude_groups(
        x,
        segment.even_positions,
        segment.even_magnitudes,
        segment.even_counts,
    )

    ordered_positions = []
    ordered_magnitudes = []
    for magnitude, positions in zip(segment.odd_magnitudes, odd_groups, strict=True):
        ordered_positions.extend(positions)
        ordered_magnitudes.extend([magnitude] * len(positions))
    for magnitude, positions in zip(segment.even_magnitudes, even_groups, strict=True):
        ordered_positions.extend(positions)
        ordered_magnitudes.extend([magnitude] * len(positions))

    signs = _best_valid_signs(x, ordered_positions, tuple(ordered_magnitudes))
    vector = torch.zeros(DIM, dtype=torch.float32)
    for pos, magnitude, sign in zip(ordered_positions, ordered_magnitudes, signs, strict=True):
        vector[pos] = float(sign * magnitude)

    odd_rank = _rank_group_assignment(segment.odd_positions, segment.odd_counts, odd_groups)
    even_rank = _rank_group_assignment(segment.even_positions, segment.even_counts, even_groups)
    sign_rank = _sign_rank(tuple(ordered_magnitudes), signs)
    place_rank = odd_rank * segment.even_place_count + even_rank
    index = segment.global_start + place_rank * segment.sign_count + sign_rank
    return vector, index


def _greedy_magnitude_groups(
    x: torch.Tensor,
    positions: tuple[int, ...],
    magnitudes: tuple[int, ...],
    counts: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    groups_by_magnitude: dict[int, tuple[int, ...]] = {magnitude: () for magnitude in magnitudes}
    ranked_positions = sorted(positions, key=lambda pos: (abs(float(x[pos])), -pos), reverse=True)
    cursor = 0

    for magnitude, count in sorted(zip(magnitudes, counts, strict=True), reverse=True):
        chosen = tuple(sorted(ranked_positions[cursor : cursor + count]))
        groups_by_magnitude[magnitude] = chosen
        cursor += count

    return tuple(groups_by_magnitude[magnitude] for magnitude in magnitudes)


def _best_valid_signs(
    x: torch.Tensor,
    positions: list[int],
    magnitudes: tuple[int, ...],
) -> tuple[int, ...]:
    if not magnitudes:
        return ()

    # dp[residue] = (score, signs_so_far)
    dp: list[tuple[float, tuple[int, ...]] | None] = [(0.0, ())] + [None, None, None]
    for pos, magnitude in zip(positions, magnitudes, strict=True):
        next_dp: list[tuple[float, tuple[int, ...]] | None] = [None, None, None, None]
        for residue, state in enumerate(dp):
            if state is None:
                continue
            score, signs = state
            for sign in (-1, 1):
                next_residue = (residue + sign * magnitude) % 4
                next_score = score + sign * magnitude * float(x[pos])
                old = next_dp[next_residue]
                if old is None or next_score > old[0]:
                    next_dp[next_residue] = (next_score, (*signs, sign))
        dp = next_dp

    if dp[0] is None:
        raise RuntimeError("no valid sign pattern for segment")
    return dp[0][1]


def _score(x: torch.Tensor, candidate: torch.Tensor, mode: str, x_norm: float) -> float:
    if mode == "euclidean":
        return float((x - candidate).square().sum().item())

    cand_norm = float(torch.linalg.norm(candidate).item())
    denom = max(x_norm * cand_norm, 1.0e-12)
    cosine = float(torch.dot(x, candidate).item()) / denom
    return 1.0 - cosine


def _rank_group_assignment(
    items: tuple[int, ...],
    counts: tuple[int, ...],
    groups: tuple[tuple[int, ...], ...],
) -> int:
    rank = 0
    remaining_items = tuple(items)

    for group_idx, (group_count, group) in enumerate(zip(counts, groups, strict=True)):
        suffix_counts = counts[group_idx + 1 :]
        suffix_count = _group_assignment_count(len(remaining_items) - group_count, suffix_counts)
        combo_rank = _rank_combination(remaining_items, tuple(sorted(group)))
        rank += combo_rank * suffix_count
        selected = set(group)
        remaining_items = tuple(item for item in remaining_items if item not in selected)

    return rank


def _group_assignment_count(total_positions: int, counts: tuple[int, ...]) -> int:
    if sum(counts) > total_positions:
        return 0

    count = 1
    remaining = total_positions
    for group_count in counts:
        count *= comb(remaining, group_count)
        remaining -= group_count
    return count


def _rank_combination(items: tuple[int, ...], chosen: tuple[int, ...]) -> int:
    if not chosen:
        return 0

    rank = 0
    start = 0
    n = len(items)
    k = len(chosen)
    chosen_indices = [items.index(item) for item in chosen]

    for selected_idx, item_idx in enumerate(chosen_indices):
        remaining = k - selected_idx
        for skipped in range(start, item_idx):
            rank += comb(n - skipped - 1, remaining - 1)
        start = item_idx + 1

    return rank


def _sign_rank(magnitudes: tuple[int, ...], signs: tuple[int, ...]) -> int:
    patterns = _valid_sign_patterns(magnitudes)
    target = torch.tensor(signs, dtype=torch.int16)
    matches = torch.nonzero((patterns == target).all(dim=1), as_tuple=False).flatten()
    if matches.numel() == 0:
        raise RuntimeError("sign pattern was not valid")
    return int(matches[0].item())


def demo(
    M: int = 8,
    max_coord: int = 4,
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
    db = ImplicitShellDatabase(device=device)
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

    parser = argparse.ArgumentParser(description="Adoul-Barth-style LLVQ structured search demo")
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--max_coord", type=int, default=4)
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
