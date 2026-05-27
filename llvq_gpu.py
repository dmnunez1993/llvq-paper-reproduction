from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

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
        eye = torch.eye(12, dtype=torch.int16, device=device)
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
            dtype=torch.int16,
            device=device,
        )
        return torch.cat([eye, parity], dim=1)

    def encode(self, msg: torch.Tensor) -> torch.Tensor:
        msg = msg.to(device=self.device, dtype=torch.float32)
        generator = self.G.to(torch.float32)
        return ((msg @ generator).remainder(2)).to(torch.int16)

    def _generate_codewords(self) -> torch.Tensor:
        ids = torch.arange(1 << 12, device=self.device, dtype=torch.int64)
        shifts = torch.arange(12, device=self.device, dtype=torch.int64)
        messages = ((ids[:, None] >> shifts[None, :]) & 1).to(torch.int16)
        return self.encode(messages)


@dataclass
class Shell:
    m: int
    vectors: torch.Tensor


class ShellDatabase:
    """
    Finite Leech-lattice-like shell database using Construction A constraints.

    The generator keeps the combinatorial recursion bounded by norm and uses Torch
    batches on the selected device to filter/pack candidates into shells.
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
        max_coord: int = 1,
        batch_size: int = 262_144,
        max_vectors_per_shell: int | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Build shells with squared norm 2m for 2 <= m <= M.

        `max_coord=1` is a practical default for early shells. Larger bounds can
        grow quickly because the Leech lattice has very large shell populations.
        """

        if M < 2:
            raise ValueError("M must be at least 2")
        if max_coord < 1:
            raise ValueError("max_coord must be at least 1")

        golay = GolayCode(self.device)
        shell_map: dict[int, list[torch.Tensor]] = defaultdict(list)
        shell_counts: dict[int, int] = defaultdict(int)
        max_norm2 = 2 * M
        allowed_even = tuple(v for v in range(-max_coord, max_coord + 1) if v % 2 == 0)
        allowed_odd = tuple(v for v in range(-max_coord, max_coord + 1) if v % 2 != 0)

        if verbose:
            print(
                f"building shells on {self.device} "
                f"(M={M}, max_coord={max_coord}, codewords={golay.codewords.shape[0]})",
                flush=True,
            )

        for codeword_idx, codeword in enumerate(golay.codewords.cpu().tolist(), start=1):
            values_by_coord = [allowed_odd if bit else allowed_even for bit in codeword]
            codeword_candidates = 0
            for batch in _bounded_candidate_batches(values_by_coord, max_norm2, batch_size):
                codeword_candidates += len(batch)
                x = torch.tensor(batch, dtype=torch.int16, device=self.device)
                norm2 = (x.to(torch.int32) * x.to(torch.int32)).sum(dim=1)
                valid = (norm2 >= 4) & (norm2 <= max_norm2) & ((norm2 % 2) == 0)
                valid &= (x.to(torch.int32).sum(dim=1) % 4) == 0
                x = x[valid]
                norm2 = norm2[valid]

                for m in torch.unique(norm2 // 2).tolist():
                    shell_vectors = x[(norm2 // 2) == m].to(self.dtype)
                    if max_vectors_per_shell is not None:
                        remaining = max_vectors_per_shell - shell_counts[m]
                        if remaining <= 0:
                            continue
                        shell_vectors = shell_vectors[:remaining]
                    if shell_vectors.numel() == 0:
                        continue
                    shell_map[m].append(shell_vectors)
                    shell_counts[m] += shell_vectors.shape[0]

            if verbose and (
                codeword_idx == 1
                or codeword_idx % 128 == 0
                or codeword_idx == golay.codewords.shape[0]
            ):
                print(
                    f"  codeword {codeword_idx:4d}/{golay.codewords.shape[0]} | "
                    f"candidates={codeword_candidates:8d} | "
                    f"vectors={sum(shell_counts.values()):8d} | "
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


def _bounded_candidate_batches(
    values_by_coord: list[tuple[int, ...]],
    max_norm2: int,
    batch_size: int,
) -> Iterable[list[list[int]]]:
    min_suffix = [0] * (DIM + 1)
    for i in range(DIM - 1, -1, -1):
        min_suffix[i] = min_suffix[i + 1] + min(v * v for v in values_by_coord[i])

    batch: list[list[int]] = []
    current = [0] * DIM

    def visit(pos: int, norm2: int) -> Iterable[list[list[int]]]:
        if norm2 + min_suffix[pos] > max_norm2:
            return
        if pos == DIM:
            batch.append(current.copy())
            if len(batch) >= batch_size:
                out = batch.copy()
                batch.clear()
                yield out
            return

        for value in values_by_coord[pos]:
            next_norm2 = norm2 + value * value
            if next_norm2 + min_suffix[pos + 1] <= max_norm2:
                current[pos] = value
                yield from visit(pos + 1, next_norm2)

    yield from visit(0, 0)
    if batch:
        yield batch


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
        """
        Quantize one or more 24-D vectors.

        Returns `(indices, vectors, scores)` by default. For a single input vector,
        the returned index and score are scalar tensors.
        """

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


if __name__ == "__main__":
    db = ShellDatabase()
    db.build(M=8, max_coord=2, verbose=True)
    print("device:", db.device)
    print("shell sizes:", db.shell_sizes())
    print("total vectors:", db.total_count)

    quantizer = LLVQ(db)
    x = torch.randn(8, DIM, device=db.device) * 3.0
    idx, vectors, scores = quantizer.quantize(x)
    recon = quantizer.dequantize(idx)
    diff = x - recon

    print("indices:", idx)
    print("scores:", scores)
    print("x:", x)
    print("recon:", recon)
    print("x - recon:", diff)
    print("per-vector l2 error:", diff.norm(dim=1))
    print("mean squared error:", diff.square().mean())
    print("reconstruction matches:", torch.equal(vectors, recon))
