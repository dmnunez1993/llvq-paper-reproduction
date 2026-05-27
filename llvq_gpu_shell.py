from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import torch


DIM = 24


def _default_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@dataclass
class Shell:
    m: int
    vectors: torch.Tensor


class ShellDatabase:
    """
    GPU-expanded shell database for the practical demo case `max_coord <= 2`.

    Compared with `llvq_gpu.py`, this avoids Python recursion over 24 coordinates.
    Python still enumerates small support choices, but each block of sign patterns
    is expanded, filtered, and stored as Torch tensors on the GPU.
    """

    def __init__(self, device: str | torch.device | None = None, dtype: torch.dtype = torch.float32):
        self.device = _default_device(device)
        self.dtype = dtype
        self.shells: dict[int, Shell] = {}
        self.offsets: dict[int, int] = {}
        self.all_vectors = torch.empty((0, DIM), dtype=dtype, device=self.device)
        self.total_count = 0
        self._index: dict[tuple[int, ...], int] = {}

    def build(
        self,
        M: int,
        max_coord: int = 2,
        max_vectors_per_shell: int | None = None,
        verbose: bool = False,
    ) -> None:
        if M < 2:
            raise ValueError("M must be at least 2")
        if max_coord > 2:
            raise ValueError("llvq_gpu_shell.py currently supports max_coord <= 2")
        if max_coord < 1:
            raise ValueError("max_coord must be at least 1")

        golay = GolayCode(self.device)
        shell_map: dict[int, list[torch.Tensor]] = defaultdict(list)
        shell_counts: dict[int, int] = defaultdict(int)
        max_norm2 = 2 * M

        if verbose:
            print(
                f"gpu shell build on {self.device} "
                f"(M={M}, max_coord={max_coord}, codewords={golay.codewords.shape[0]})",
                flush=True,
            )

        codewords = golay.codewords.cpu()
        for codeword_idx, codeword in enumerate(codewords, start=1):
            odd_positions = torch.nonzero(codeword == 1, as_tuple=False).flatten().tolist()
            even_positions = torch.nonzero(codeword == 0, as_tuple=False).flatten().tolist()
            odd_count = len(odd_positions)

            if odd_count > max_norm2:
                if verbose and self._should_report(codeword_idx, len(codewords)):
                    print(
                        f"  codeword {codeword_idx:4d}/{len(codewords)} | skipped | "
                        f"vectors={sum(shell_counts.values()):8d}",
                        flush=True,
                    )
                continue

            odd_signs = _sign_patterns(odd_count, self.device)
            max_twos = 0 if max_coord < 2 else (max_norm2 - odd_count) // 4
            codeword_added = 0

            for two_count in range(max_twos + 1):
                norm2_value = odd_count + 4 * two_count
                if norm2_value < 4 or norm2_value > max_norm2 or norm2_value % 2 != 0:
                    continue
                m = norm2_value // 2
                if max_vectors_per_shell is not None and shell_counts[m] >= max_vectors_per_shell:
                    continue

                for two_positions in combinations(even_positions, two_count):
                    two_signs = _sign_patterns(two_count, self.device)
                    vectors = _expand_vectors(odd_positions, odd_signs, two_positions, two_signs, self.device)
                    if vectors.numel() == 0:
                        continue

                    valid = (vectors.to(torch.int32).sum(dim=1) % 4) == 0
                    vectors = vectors[valid]
                    if vectors.numel() == 0:
                        continue

                    if max_vectors_per_shell is not None:
                        remaining = max_vectors_per_shell - shell_counts[m]
                        if remaining <= 0:
                            continue
                        vectors = vectors[:remaining]

                    vectors = vectors.to(self.dtype)
                    shell_map[m].append(vectors)
                    shell_counts[m] += vectors.shape[0]
                    codeword_added += vectors.shape[0]

            if verbose and self._should_report(codeword_idx, len(codewords)):
                print(
                    f"  codeword {codeword_idx:4d}/{len(codewords)} | "
                    f"added={codeword_added:8d} | vectors={sum(shell_counts.values()):8d} | "
                    f"shells={dict(sorted(shell_counts.items()))}",
                    flush=True,
                )

        self.shells.clear()
        self.offsets.clear()
        self._index.clear()

        chunks = []
        offset = 0
        for m in sorted(shell_map):
            vectors = torch.cat(shell_map[m], dim=0)
            self.shells[m] = Shell(m=m, vectors=vectors)
            self.offsets[m] = offset
            offset += vectors.shape[0]
            chunks.append(vectors)

        self.all_vectors = torch.cat(chunks, dim=0) if chunks else torch.empty(
            (0, DIM), dtype=self.dtype, device=self.device
        )
        self.total_count = self.all_vectors.shape[0]
        self._rebuild_index()

        if verbose:
            print(
                f"finished: total_vectors={self.total_count}, shell_sizes={self.shell_sizes()}",
                flush=True,
            )

    @staticmethod
    def _should_report(codeword_idx: int, total_codewords: int) -> bool:
        return codeword_idx == 1 or codeword_idx % 128 == 0 or codeword_idx == total_codewords

    def shell_sizes(self) -> dict[int, int]:
        return {m: shell.vectors.shape[0] for m, shell in self.shells.items()}

    def to(self, device: str | torch.device) -> ShellDatabase:
        device = torch.device(device)
        self.device = device
        self.all_vectors = self.all_vectors.to(device)
        for shell in self.shells.values():
            shell.vectors = shell.vectors.to(device)
        self._rebuild_index()
        return self

    def _rebuild_index(self) -> None:
        vectors = self.all_vectors.to("cpu", dtype=torch.int64)
        self._index = {tuple(v.tolist()): i for i, v in enumerate(vectors)}


def _sign_patterns(width: int, device: torch.device) -> torch.Tensor:
    if width == 0:
        return torch.empty((1, 0), dtype=torch.int16, device=device)

    ids = torch.arange(1 << width, dtype=torch.int64, device=device)
    shifts = torch.arange(width, dtype=torch.int64, device=device)
    bits = ((ids[:, None] >> shifts[None, :]) & 1).to(torch.int16)
    return bits * 2 - 1


def _expand_vectors(
    odd_positions: list[int],
    odd_signs: torch.Tensor,
    two_positions: tuple[int, ...],
    two_signs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    odd_count = len(odd_positions)
    two_count = len(two_positions)
    total = odd_signs.shape[0] * two_signs.shape[0]
    vectors = torch.zeros((total, DIM), dtype=torch.int16, device=device)

    if odd_count:
        odd_idx = torch.tensor(odd_positions, dtype=torch.long, device=device)
        repeated_odd_signs = odd_signs.repeat_interleave(two_signs.shape[0], dim=0)
        vectors[:, odd_idx] = repeated_odd_signs

    if two_count:
        two_idx = torch.tensor(two_positions, dtype=torch.long, device=device)
        repeated_two_signs = two_signs.repeat(odd_signs.shape[0], 1)
        vectors[:, two_idx] = repeated_two_signs * 2

    return vectors


class LLVQ:
    """Torch quantizer/dequantizer over a finite shell database."""

    def __init__(self, shell_db: ShellDatabase, search_chunk_size: int = 65_536):
        if shell_db.total_count == 0:
            raise ValueError("shell_db is empty; call ShellDatabase.build(...) first")
        self.db = shell_db
        self.search_chunk_size = search_chunk_size

    def index_to_vector(self, idx: int | torch.Tensor) -> torch.Tensor:
        return self.dequantize(idx)

    def vector_to_index(self, vec: torch.Tensor) -> int:
        key = tuple(torch.as_tensor(vec, dtype=torch.int64).cpu().tolist())
        try:
            return self.db._index[key]
        except KeyError as exc:
            raise ValueError("vector not found") from exc

    def quantize(
        self,
        x: torch.Tensor,
        mode: str = "euclidean",
        return_vectors: bool = True,
    ):
        x = torch.as_tensor(x, dtype=self.db.dtype, device=self.db.device)
        single = x.ndim == 1
        x = x.reshape(-1, DIM)

        best_scores = torch.full((x.shape[0],), torch.inf, dtype=self.db.dtype, device=self.db.device)
        best_indices = torch.zeros((x.shape[0],), dtype=torch.long, device=self.db.device)

        for start in range(0, self.db.total_count, self.search_chunk_size):
            end = min(start + self.search_chunk_size, self.db.total_count)
            codebook = self.db.all_vectors[start:end]

            if mode == "euclidean":
                scores = torch.cdist(x, codebook, p=2).square()
            elif mode == "angular":
                x_norm = torch.nn.functional.normalize(x, dim=1)
                v_norm = torch.nn.functional.normalize(codebook, dim=1)
                scores = 1.0 - x_norm @ v_norm.T
            else:
                raise ValueError(f"unknown quantization mode: {mode}")

            chunk_scores, chunk_indices = scores.min(dim=1)
            improved = chunk_scores < best_scores
            best_scores[improved] = chunk_scores[improved]
            best_indices[improved] = chunk_indices[improved] + start

        vectors = self.dequantize(best_indices) if return_vectors else None

        if single:
            best_indices = best_indices[0]
            best_scores = best_scores[0]
            vectors = vectors[0] if vectors is not None else None

        return best_indices, vectors, best_scores

    def dequantize(self, idx: int | torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.db.device)
        return self.db.all_vectors[idx]


def demo(
    M: int = 5,
    max_coord: int = 2,
    batch_vectors: int = 8,
    input_scale: float = 3.0,
    mode: str = "euclidean",
    max_vectors_per_shell: int | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> None:
    db = ShellDatabase(device=device)
    db.build(
        M=M,
        max_coord=max_coord,
        max_vectors_per_shell=max_vectors_per_shell,
        verbose=verbose,
    )
    print("device:", db.device)
    print("shell sizes:", db.shell_sizes())
    print("total vectors:", db.total_count)

    quantizer = LLVQ(db)
    x = torch.randn(batch_vectors, DIM, device=db.device) * input_scale
    idx, recon, scores = quantizer.quantize(x, mode=mode)
    diff = x - recon

    print("indices:", idx)
    print("scores:", scores)
    print("x:", x)
    print("recon:", recon)
    print("x - recon:", diff)
    print("per-vector l2 error:", diff.norm(dim=1))
    print("mean squared error:", diff.square().mean())


def _argparse_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="GPU shell LLVQ demo")
    parser.add_argument("--M", type=int, default=5)
    parser.add_argument("--max_coord", type=int, default=2)
    parser.add_argument("--batch_vectors", type=int, default=8)
    parser.add_argument("--input_scale", type=float, default=3.0)
    parser.add_argument("--mode", choices=["euclidean", "angular"], default="euclidean")
    parser.add_argument("--max_vectors_per_shell", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    demo(
        M=args.M,
        max_coord=args.max_coord,
        batch_vectors=args.batch_vectors,
        input_scale=args.input_scale,
        mode=args.mode,
        max_vectors_per_shell=args.max_vectors_per_shell,
        device=args.device,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        _argparse_main()
    else:
        fire.Fire(demo)
