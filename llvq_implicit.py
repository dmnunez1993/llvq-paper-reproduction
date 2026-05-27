from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import ceil, comb, log2
from pathlib import Path
from time import perf_counter

import torch


DIM = 24


def _default_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cache_path(cache_dir: str | Path | None, M: int, max_coord: int) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"implicit_segments_v1_M{M}_C{max_coord}.pt"


class GolayCode:
    """Extended binary Golay code (24, 12, 8), represented as Torch tensors."""

    def __init__(self, device: str | torch.device | None = None):
        self.device = _default_device(device)
        self.G = self._generator_matrix(self.device)
        self.codewords = self._generate_codewords()

    @staticmethod
    def _generator_matrix(device: torch.device) -> torch.Tensor:
        eye = torch.eye(12, dtype=torch.float32, device=device)
        parity = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        return torch.cat([eye, parity], dim=1)

    def encode(self, msg: torch.Tensor) -> torch.Tensor:
        msg = msg.to(device=self.device, dtype=torch.float32)
        return ((msg @ self.G).remainder(2)).to(torch.int16)

    def _generate_codewords(self) -> torch.Tensor:
        ids = torch.arange(1 << 12, device=self.device, dtype=torch.int64)
        shifts = torch.arange(12, device=self.device, dtype=torch.int64)
        messages = ((ids[:, None] >> shifts[None, :]) & 1).to(torch.float32)
        return self.encode(messages)


@dataclass(frozen=True)
class Segment:
    shell: int
    codeword_idx: int
    odd_positions: tuple[int, ...]
    even_positions: tuple[int, ...]
    odd_magnitudes: tuple[int, ...]
    even_magnitudes: tuple[int, ...]
    odd_counts: tuple[int, ...]
    even_counts: tuple[int, ...]
    odd_place_count: int
    even_place_count: int
    sign_count: int
    count: int
    shell_start: int
    global_start: int


class ImplicitShellDatabase:
    """
    Codebook-free shell database for bounded integer-coordinate Leech vectors.

    It stores compact segment metadata only. Vectors are reconstructed from an
    integer index by unranking: shell -> Golay codeword -> magnitude pattern ->
    coordinate placements -> valid sign pattern. Quantization streams generated
    candidates but does not keep a full `N x 24` codebook tensor in memory.
    """

    def __init__(self, device: str | torch.device | None = None, dtype: torch.dtype = torch.float32):
        self.device = _default_device(device)
        self.dtype = dtype
        self.M = 0
        self.max_coord = 0
        self.shell_offsets: dict[int, int] = {}
        self.shell_counts: dict[int, int] = {}
        self.segments: list[Segment] = []
        self.total_count = 0
        self.golay = GolayCode(self.device)

    def build(
        self,
        M: int,
        max_coord: int = 2,
        verbose: bool = False,
        progress_every: int = 128,
        cache_dir: str | Path | None = ".llvq_cache",
        use_cache: bool = True,
    ) -> None:
        if M < 2:
            raise ValueError("M must be at least 2")
        if max_coord < 1:
            raise ValueError("max_coord must be at least 1")

        cache_path = _cache_path(cache_dir, M, max_coord)
        if use_cache and cache_path is not None and cache_path.exists():
            if verbose:
                print(f"loading implicit build cache: {cache_path}", flush=True)
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

        pending: dict[
            int,
            list[
                tuple[
                    int,
                    tuple[int, ...],
                    tuple[int, ...],
                    tuple[int, ...],
                    tuple[int, ...],
                    tuple[int, ...],
                    tuple[int, ...],
                    int,
                    int,
                    int,
                    int,
                ]
            ],
        ] = {}
        progress_shell_counts: dict[int, int] = {}
        progress_segments = 0
        progress_candidates = 0
        max_norm2 = 2 * M
        odd_magnitudes = tuple(range(1, max_coord + 1, 2))
        even_magnitudes = tuple(range(2, max_coord + 1, 2))

        if verbose:
            print(f"implicit build on {self.device} (M={M}, max_coord={max_coord})", flush=True)

        codewords = self.golay.codewords.cpu().tolist()
        for codeword_idx, codeword in enumerate(codewords, start=1):
            odd_positions = tuple(i for i, bit in enumerate(codeword) if bit)
            even_positions = tuple(i for i, bit in enumerate(codeword) if not bit)
            odd_count = len(odd_positions)
            codeword_segments = 0
            codeword_candidates = 0

            for odd_counts in _count_compositions(odd_count, len(odd_magnitudes)):
                odd_norm2 = sum(c * (m * m) for c, m in zip(odd_counts, odd_magnitudes, strict=True))
                if odd_norm2 > max_norm2:
                    continue

                even_budget = max_norm2 - odd_norm2
                for even_counts in _bounded_counts(len(even_positions), even_magnitudes, even_budget):
                    norm2 = odd_norm2 + sum(
                        c * (m * m) for c, m in zip(even_counts, even_magnitudes, strict=True)
                    )
                    if norm2 < 4 or norm2 > max_norm2 or norm2 % 2:
                        continue

                    sign_magnitudes = _expanded_magnitudes(odd_magnitudes, odd_counts)
                    sign_magnitudes += _expanded_magnitudes(even_magnitudes, even_counts)
                    sign_count = _valid_sign_count(sign_magnitudes)
                    if sign_count == 0:
                        continue

                    odd_place_count = _group_assignment_count(len(odd_positions), odd_counts)
                    even_place_count = _group_assignment_count(len(even_positions), even_counts)
                    count = odd_place_count * even_place_count * sign_count
                    shell = norm2 // 2
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
                or codeword_idx == len(codewords)
            ):
                pct = 100.0 * codeword_idx / len(codewords)
                print(
                    f"  build codeword {codeword_idx:4d}/{len(codewords)} ({pct:6.2f}%) | "
                    f"added_segments={codeword_segments:7d} | "
                    f"added_candidates={codeword_candidates:14d} | "
                    f"segments={progress_segments:9d} | "
                    f"candidates={progress_candidates:14d} | "
                    f"shells={dict(sorted(progress_shell_counts.items()))}",
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
                    odd_magnitudes,
                    even_magnitudes,
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
                        odd_magnitudes=odd_magnitudes,
                        even_magnitudes=even_magnitudes,
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
            print(f"finished: total_vectors={self.total_count}, shell_sizes={self.shell_counts}", flush=True)

        if use_cache and cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "version": 1,
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
                print(f"saved implicit build cache: {cache_path}", flush=True)

    def shell_sizes(self) -> dict[int, int]:
        return dict(self.shell_counts)

    def dequantize(self, idx: int | torch.Tensor) -> torch.Tensor:
        idx_tensor = torch.as_tensor(idx, dtype=torch.long)
        single = idx_tensor.ndim == 0
        flat = idx_tensor.reshape(-1).cpu().tolist()
        vectors = [self._decode_one(int(i)) for i in flat]
        out = torch.stack(vectors, dim=0).to(device=self.device, dtype=self.dtype)
        return out[0] if single else out.reshape(*idx_tensor.shape, DIM)

    def _decode_one(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.total_count:
            raise IndexError(idx)

        segment = self._segment_for_index(idx)
        local = idx - segment.global_start
        sign_rank = local % segment.sign_count
        place_rank = local // segment.sign_count
        even_place_rank = place_rank % segment.even_place_count
        odd_place_rank = place_rank // segment.even_place_count

        odd_groups = _unrank_group_assignment(segment.odd_positions, segment.odd_counts, odd_place_rank)
        even_groups = _unrank_group_assignment(segment.even_positions, segment.even_counts, even_place_rank)
        sign_magnitudes = _expanded_magnitudes(segment.odd_magnitudes, segment.odd_counts)
        sign_magnitudes += _expanded_magnitudes(segment.even_magnitudes, segment.even_counts)
        signs = _valid_sign_patterns(sign_magnitudes)[sign_rank]

        vector = torch.zeros(DIM, dtype=torch.int16)
        cursor = 0
        for magnitude, positions in zip(segment.odd_magnitudes, odd_groups, strict=True):
            for pos in positions:
                vector[pos] = int(signs[cursor]) * magnitude
                cursor += 1
        for magnitude, positions in zip(segment.even_magnitudes, even_groups, strict=True):
            for pos in positions:
                vector[pos] = int(signs[cursor]) * magnitude
                cursor += 1
        return vector

    def _segment_for_index(self, idx: int) -> Segment:
        lo = 0
        hi = len(self.segments)
        while lo < hi:
            mid = (lo + hi) // 2
            segment = self.segments[mid]
            if idx < segment.global_start:
                hi = mid
            elif idx >= segment.global_start + segment.count:
                lo = mid + 1
            else:
                return segment
        raise IndexError(idx)

    def iter_candidate_chunks(self, chunk_size: int = 65_536):
        for segment in self.segments:
            emitted = 0
            sign_magnitudes = _expanded_magnitudes(segment.odd_magnitudes, segment.odd_counts)
            sign_magnitudes += _expanded_magnitudes(segment.even_magnitudes, segment.even_counts)
            patterns = _valid_sign_patterns(sign_magnitudes)

            for odd_place_rank in range(segment.odd_place_count):
                odd_groups = _unrank_group_assignment(segment.odd_positions, segment.odd_counts, odd_place_rank)
                for even_place_rank in range(segment.even_place_count):
                    even_groups = _unrank_group_assignment(segment.even_positions, segment.even_counts, even_place_rank)
                    vectors = _vectors_from_patterns(
                        segment.odd_magnitudes,
                        odd_groups,
                        segment.even_magnitudes,
                        even_groups,
                        patterns,
                        self.device,
                    )
                    place_rank = odd_place_rank * segment.even_place_count + even_place_rank
                    base = segment.global_start + place_rank * segment.sign_count

                    for start in range(0, vectors.shape[0], chunk_size):
                        end = min(start + chunk_size, vectors.shape[0])
                        yield base + start, vectors[start:end].to(self.dtype)
                        emitted += end - start

            if emitted != segment.count:
                raise RuntimeError("internal indexing/count mismatch")


@lru_cache(maxsize=None)
def _valid_sign_count(magnitudes: tuple[int, ...]) -> int:
    return int(_valid_sign_patterns(magnitudes).shape[0])


@lru_cache(maxsize=None)
def _valid_sign_patterns(magnitudes: tuple[int, ...]) -> torch.Tensor:
    width = len(magnitudes)
    if width == 0:
        return torch.empty((0, 0), dtype=torch.int16)
    if width > 24:
        raise ValueError("too many nonzero coordinates for exhaustive sign pattern enumeration")

    ids = torch.arange(1 << width, dtype=torch.int64)
    shifts = torch.arange(width, dtype=torch.int64)
    bits = ((ids[:, None] >> shifts[None, :]) & 1).to(torch.int16)
    signs = bits * 2 - 1
    mags = torch.tensor(magnitudes, dtype=torch.int16)
    return signs[((signs * mags).sum(dim=1) % 4) == 0].contiguous()


def _vectors_from_patterns(
    odd_magnitudes: tuple[int, ...],
    odd_groups: tuple[tuple[int, ...], ...],
    even_magnitudes: tuple[int, ...],
    even_groups: tuple[tuple[int, ...], ...],
    patterns: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    vectors = torch.zeros((patterns.shape[0], DIM), dtype=torch.int16, device=device)
    patterns = patterns.to(device)
    cursor = 0

    for magnitude, positions in zip(odd_magnitudes, odd_groups, strict=True):
        if positions:
            idx = torch.tensor(positions, dtype=torch.long, device=device)
            width = len(positions)
            vectors[:, idx] = patterns[:, cursor : cursor + width] * magnitude
            cursor += width

    for magnitude, positions in zip(even_magnitudes, even_groups, strict=True):
        if positions:
            idx = torch.tensor(positions, dtype=torch.long, device=device)
            width = len(positions)
            vectors[:, idx] = patterns[:, cursor : cursor + width] * magnitude
            cursor += width

    return vectors


@lru_cache(maxsize=None)
def _count_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts <= 0:
        return ((),) if total == 0 else ()
    if parts == 1:
        return ((total,),)

    out = []
    for first in range(total + 1):
        for rest in _count_compositions(total - first, parts - 1):
            out.append((first, *rest))
    return tuple(out)


@lru_cache(maxsize=None)
def _bounded_counts(total_positions: int, magnitudes: tuple[int, ...], max_norm2: int) -> tuple[tuple[int, ...], ...]:
    if not magnitudes:
        return ((),)

    out = []
    current = [0] * len(magnitudes)

    def visit(pos: int, remaining_positions: int, remaining_norm2: int) -> None:
        if pos == len(magnitudes):
            out.append(tuple(current))
            return

        magnitude_norm2 = magnitudes[pos] * magnitudes[pos]
        max_count = min(remaining_positions, remaining_norm2 // magnitude_norm2)
        for count in range(max_count + 1):
            current[pos] = count
            visit(pos + 1, remaining_positions - count, remaining_norm2 - count * magnitude_norm2)

    visit(0, total_positions, max_norm2)
    return tuple(out)


@lru_cache(maxsize=None)
def _expanded_magnitudes(magnitudes: tuple[int, ...], counts: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for magnitude, count in zip(magnitudes, counts, strict=True):
        out.extend([magnitude] * count)
    return tuple(out)


@lru_cache(maxsize=None)
def _group_assignment_count(total_positions: int, counts: tuple[int, ...]) -> int:
    if sum(counts) > total_positions:
        return 0

    count = 1
    remaining = total_positions
    for group_count in counts:
        count *= comb(remaining, group_count)
        remaining -= group_count
    return count


def _unrank_group_assignment(
    items: tuple[int, ...],
    counts: tuple[int, ...],
    rank: int,
) -> tuple[tuple[int, ...], ...]:
    total = _group_assignment_count(len(items), counts)
    if rank < 0 or rank >= total:
        raise IndexError(rank)

    remaining_items = tuple(items)
    groups = []
    for group_idx, group_count in enumerate(counts):
        suffix_counts = counts[group_idx + 1 :]
        suffix_count = _group_assignment_count(len(remaining_items) - group_count, suffix_counts)
        combo_rank = rank // suffix_count if suffix_count else 0
        rank = rank % suffix_count if suffix_count else 0
        group = _unrank_combination(remaining_items, group_count, combo_rank)
        groups.append(group)
        selected = set(group)
        remaining_items = tuple(item for item in remaining_items if item not in selected)

    return tuple(groups)


def _unrank_combination(items: tuple[int, ...], k: int, rank: int) -> tuple[int, ...]:
    if k == 0:
        return ()
    if rank < 0 or rank >= comb(len(items), k):
        raise IndexError(rank)

    result = []
    start = 0
    n = len(items)
    for remaining in range(k, 0, -1):
        for i in range(start, n):
            count = comb(n - i - 1, remaining - 1)
            if rank < count:
                result.append(items[i])
                start = i + 1
                break
            rank -= count
    return tuple(result)


class LLVQImplicit:
    """Quantizer/dequantizer that streams implicit candidates instead of storing a codebook."""

    def __init__(self, shell_db: ImplicitShellDatabase, search_chunk_size: int = 65_536):
        if shell_db.total_count == 0:
            raise ValueError("shell_db is empty; call build(...) first")
        self.db = shell_db
        self.search_chunk_size = search_chunk_size

    def quantize(
        self,
        x: torch.Tensor,
        mode: str = "euclidean",
        return_vectors: bool = True,
        verbose: bool = False,
        progress_every: int = 1_000_000,
    ):
        x = torch.as_tensor(x, dtype=self.db.dtype, device=self.db.device)
        single = x.ndim == 1
        x = x.reshape(-1, DIM)

        best_scores = torch.full((x.shape[0],), torch.inf, dtype=self.db.dtype, device=self.db.device)
        best_indices = torch.zeros((x.shape[0],), dtype=torch.long, device=self.db.device)

        if mode == "angular":
            x_for_score = torch.nn.functional.normalize(x, dim=1)
        elif mode == "euclidean":
            x_for_score = x
        else:
            raise ValueError(f"unknown quantization mode: {mode}")

        seen = 0
        next_report = progress_every
        total = self.db.total_count

        if verbose:
            print(
                f"quantizing {x.shape[0]} vector(s) over {total} implicit candidates "
                f"(mode={mode}, chunk_size={self.search_chunk_size})",
                flush=True,
            )

        for base_idx, codebook in self.db.iter_candidate_chunks(self.search_chunk_size):
            codebook = codebook.to(device=self.db.device, dtype=self.db.dtype)
            if mode == "euclidean":
                scores = torch.cdist(x_for_score, codebook, p=2).square()
            else:
                v_norm = torch.nn.functional.normalize(codebook, dim=1)
                scores = 1.0 - x_for_score @ v_norm.T

            chunk_scores, chunk_indices = scores.min(dim=1)
            improved = chunk_scores < best_scores
            best_scores[improved] = chunk_scores[improved]
            best_indices[improved] = chunk_indices[improved] + base_idx
            seen += codebook.shape[0]

            if verbose and (seen >= next_report or seen == total):
                pct = 100.0 * seen / total
                best = best_scores.detach().min().item()
                print(
                    f"  scored {seen:12d}/{total} candidates ({pct:6.2f}%) | "
                    f"best_score={best:.6g}",
                    flush=True,
                )
                while next_report <= seen:
                    next_report += progress_every

        vectors = self.dequantize(best_indices) if return_vectors else None
        if single:
            best_indices = best_indices[0]
            best_scores = best_scores[0]
            vectors = vectors[0] if vectors is not None else None
        return best_indices, vectors, best_scores

    def dequantize(self, idx: int | torch.Tensor) -> torch.Tensor:
        return self.db.dequantize(idx)


def _tensor_nbytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def demo(
    M: int = 5,
    max_coord: int = 2,
    batch_vectors: int = 4,
    input_scale: float = 3.0,
    mode: str = "angular",
    device: str | None = None,
    verbose: bool = True,
    progress_every: int = 1_000_000,
    cache_dir: str | None = ".llvq_cache",
    no_cache: bool = False,
) -> None:
    db = ImplicitShellDatabase(device=device)
    db.build(
        M=M,
        max_coord=max_coord,
        verbose=verbose,
        cache_dir=cache_dir,
        use_cache=not no_cache,
    )
    print("device:", db.device)
    print("shell sizes:", db.shell_sizes())
    print("total implicit vectors:", db.total_count)
    print("stored dense codebook:", hasattr(db, "all_vectors"))

    quantizer = LLVQImplicit(db)
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

    print(f"quantize time: {quantize_seconds:.6f}s")
    print(f"dequantize time: {dequantize_seconds:.6f}s")
    print("timed dequantize matches:", torch.equal(recon, timed_recon))
    index_bits = ceil(log2(db.total_count)) if db.total_count > 1 else 1
    packed_index_bytes = ceil(index_bits * idx.numel() / 8)
    print(f"original x shape: {tuple(x.shape)}, bytes: {_tensor_nbytes(x)}")
    print(f"index shape: {tuple(idx.shape)}, tensor bytes: {_tensor_nbytes(idx)}")
    print(
        f"packed index size: {index_bits} bits/vector, "
        f"{packed_index_bytes} bytes for {idx.numel()} vector(s)"
    )
    print("indices:", idx)
    print("scores:", scores)
    print("x:", x)
    print("recon:", recon)
    print("x - recon:", diff)
    print("per-vector l2 error:", diff.norm(dim=1))
    print("mean squared error:", diff.square().mean())


def _argparse_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Implicit LLVQ demo without dense codebook storage")
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--max_coord", type=int, default=2)
    parser.add_argument("--batch_vectors", type=int, default=4)
    parser.add_argument("--input_scale", type=float, default=3.0)
    parser.add_argument("--mode", choices=["euclidean", "angular"], default="angular")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--progress_every", type=int, default=1_000_000)
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
