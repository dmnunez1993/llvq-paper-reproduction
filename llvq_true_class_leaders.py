from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from functools import lru_cache
from dataclasses import dataclass
from math import factorial

import torch

from llvq_true_leech import leech_shell_counts


DIM = 24
GOLAY_GENERATOR_PARITY: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0),
    (1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0),
    (1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0),
    (1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1),
    (1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0),
    (1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0),
    (0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0),
    (0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0),
    (0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0),
    (0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0),
    (0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1),
)


@dataclass(frozen=True)
class ClassLeader:
    """One true Leech class leader row from paper Table 2."""

    shell: int
    parity: str
    count: int
    # Multiplicities of absolute coordinate values.
    # `pdftotext` drops the leftmost ±8 header in Table 2; the shell-4
    # count=48 class is the [8, 0^23] leader.
    mult_abs8: int
    mult_abs6: int
    mult_abs5: int
    mult_abs4: int
    mult_abs3: int
    mult_abs2: int
    mult_abs1: int
    mult_abs0: int

    @property
    def multiplicities(self) -> dict[int, int]:
        return {
            8: self.mult_abs8,
            6: self.mult_abs6,
            5: self.mult_abs5,
            4: self.mult_abs4,
            3: self.mult_abs3,
            2: self.mult_abs2,
            1: self.mult_abs1,
            0: self.mult_abs0,
        }

    def leader(self) -> torch.Tensor:
        values: list[int] = []
        for magnitude in (8, 6, 5, 4, 3, 2, 1, 0):
            values.extend([magnitude] * self.multiplicities[magnitude])
        if len(values) != DIM:
            raise ValueError(f"class leader has {len(values)} coordinates, expected {DIM}")
        return torch.tensor(values, dtype=torch.int16)


@dataclass(frozen=True)
class RankedClass:
    global_index: int
    shell: int
    shell_local_index: int
    class_index: int
    class_local_index: int
    class_offset: int
    leader: ClassLeader


@dataclass(frozen=True)
class LocalDecomposition:
    """Local symmetry coordinates inside one class."""

    golay_choice: int
    golay_codeword: tuple[int, ...]
    sign_choice: int
    permutation_choice: int


@dataclass(frozen=True)
class ClassLocalStructure:
    """Implicit local database row for one shell/class block."""

    shell: int
    class_index: int
    leader: ClassLeader
    golay_codeword_indices: tuple[int, ...]
    sign_count: int
    permutation_count: int

    @property
    def count(self) -> int:
        return len(self.golay_codeword_indices) * self.sign_count * self.permutation_count


# Exact class rows printed in Table 2 of arXiv:2603.11021v1.
TABLE2_CLASS_LEADERS: tuple[ClassLeader, ...] = (
    # m=2
    ClassLeader(2, "even", 1104, 0, 0, 0, 2, 0, 0, 0, 22),
    ClassLeader(2, "even", 97152, 0, 0, 0, 0, 0, 8, 0, 16),
    ClassLeader(2, "odd", 98304, 0, 0, 0, 0, 1, 0, 23, 0),
    # m=3
    ClassLeader(3, "even", 3108864, 0, 0, 0, 1, 0, 8, 0, 15),
    ClassLeader(3, "even", 5275648, 0, 0, 0, 0, 0, 12, 0, 12),
    ClassLeader(3, "odd", 98304, 0, 0, 1, 0, 0, 0, 23, 0),
    ClassLeader(3, "odd", 8290304, 0, 0, 0, 0, 3, 0, 21, 0),
    # m=4
    ClassLeader(4, "even", 170016, 0, 0, 0, 4, 0, 0, 0, 20),
    ClassLeader(4, "even", 48, 1, 0, 0, 0, 0, 0, 0, 23),
    ClassLeader(4, "even", 46632960, 0, 0, 0, 2, 0, 8, 0, 14),
    ClassLeader(4, "even", 777216, 0, 1, 0, 0, 0, 7, 0, 16),
    ClassLeader(4, "even", 126615552, 0, 0, 0, 1, 0, 12, 0, 11),
    ClassLeader(4, "even", 24870912, 0, 0, 0, 0, 0, 16, 0, 8),
    ClassLeader(4, "odd", 24870912, 0, 0, 1, 0, 2, 0, 21, 0),
    ClassLeader(4, "odd", 174096384, 0, 0, 0, 0, 5, 0, 19, 0),
)


class TrueLeechClassLeaderIndex:
    """
    True class-leader rank/unrank for the Table 2 shells m=2,3,4.

    This implements the paper's class-level hierarchy:

        global index -> shell -> class leader -> class-local index

    It does not yet implement the final local-symmetry unranking inside a class
    into an actual Leech vector. That requires the full Golay refinement,
    constrained sign, and repeated-permutation rank/unrank rules for every
    class. This file is the correct foundation for that next layer.
    """

    def __init__(self, max_shell: int = 4):
        if max_shell > 4:
            raise ValueError("only Table 2 shells m=2,3,4 are available in this file")
        if max_shell < 2:
            raise ValueError("max_shell must be at least 2")

        self.max_shell = max_shell
        self.golay_codewords = extended_golay_codewords()
        self.classes_by_shell: dict[int, tuple[ClassLeader, ...]] = {
            shell: tuple(sorted(
                (row for row in TABLE2_CLASS_LEADERS if row.shell == shell),
                key=_class_sort_key,
            ))
            for shell in range(2, max_shell + 1)
        }
        self.shell_offsets = self._build_shell_offsets()
        self.class_offsets = self._build_class_offsets()
        self.local_structures = self._build_local_structures()
        self.total_count = sum(row.count for row in TABLE2_CLASS_LEADERS if row.shell <= max_shell)
        self._validate_against_theta_series()
        self._validate_local_structures()

    def rank_class(self, shell: int, class_index: int, class_local_index: int = 0) -> int:
        classes = self.classes_by_shell[shell]
        if class_index < 0 or class_index >= len(classes):
            raise IndexError(class_index)
        leader = classes[class_index]
        if class_local_index < 0 or class_local_index >= leader.count:
            raise IndexError(class_local_index)
        return self.shell_offsets[shell] + self.class_offsets[shell][class_index] + class_local_index

    def unrank_class(self, global_index: int) -> RankedClass:
        if global_index < 0 or global_index >= self.total_count:
            raise IndexError(global_index)

        shell = self._shell_for_index(global_index)
        shell_local = global_index - self.shell_offsets[shell]
        offsets = self.class_offsets[shell]
        classes = self.classes_by_shell[shell]

        class_index = 0
        for i, start in enumerate(offsets):
            end = start + classes[i].count
            if shell_local < end:
                class_index = i
                break

        class_offset = offsets[class_index]
        return RankedClass(
            global_index=global_index,
            shell=shell,
            shell_local_index=shell_local,
            class_index=class_index,
            class_local_index=shell_local - class_offset,
            class_offset=class_offset,
            leader=classes[class_index],
        )

    def leader_for_index(self, global_index: int) -> torch.Tensor:
        return self.unrank_class(global_index).leader.leader()

    def decompose_local_index(self, ranked: RankedClass) -> LocalDecomposition:
        local = self.local_structures[ranked.shell][ranked.class_index]
        if ranked.class_local_index < 0 or ranked.class_local_index >= local.count:
            raise IndexError(ranked.class_local_index)

        cursor = ranked.class_local_index
        permutation_choice = cursor % local.permutation_count
        cursor //= local.permutation_count
        sign_choice = cursor % local.sign_count
        cursor //= local.sign_count
        golay_choice = cursor
        codeword_index = local.golay_codeword_indices[golay_choice]
        return LocalDecomposition(
            golay_choice=golay_choice,
            golay_codeword=self.golay_codewords[codeword_index],
            sign_choice=sign_choice,
            permutation_choice=permutation_choice,
        )

    def dequantize(self, global_index: int) -> torch.Tensor:
        ranked = self.unrank_class(global_index)
        local = self.decompose_local_index(ranked)
        vector = unrank_class_vector(ranked.leader, local)
        return torch.tensor(vector, dtype=torch.int16)

    def _shell_for_index(self, global_index: int) -> int:
        for shell in range(2, self.max_shell + 1):
            start = self.shell_offsets[shell]
            end = start + sum(row.count for row in self.classes_by_shell[shell])
            if start <= global_index < end:
                return shell
        raise IndexError(global_index)

    def _build_shell_offsets(self) -> dict[int, int]:
        offsets = {}
        cursor = 0
        for shell in range(2, self.max_shell + 1):
            offsets[shell] = cursor
            cursor += sum(row.count for row in self.classes_by_shell[shell])
        return offsets

    def _build_class_offsets(self) -> dict[int, tuple[int, ...]]:
        out = {}
        for shell, rows in self.classes_by_shell.items():
            offsets = []
            cursor = 0
            for row in rows:
                offsets.append(cursor)
                cursor += row.count
            out[shell] = tuple(offsets)
        return out

    def _build_local_structures(self) -> dict[int, tuple[ClassLocalStructure, ...]]:
        out = {}
        for shell, rows in self.classes_by_shell.items():
            structures = []
            for class_index, row in enumerate(rows):
                structures.append(
                    build_local_structure(
                        shell=shell,
                        class_index=class_index,
                        leader=row,
                        golay_codewords=self.golay_codewords,
                    )
                )
            out[shell] = tuple(structures)
        return out

    def _validate_against_theta_series(self) -> None:
        true_counts = leech_shell_counts(self.max_shell)
        for shell, rows in self.classes_by_shell.items():
            table_count = sum(row.count for row in rows)
            if table_count != true_counts[shell]:
                raise ValueError(
                    f"Table 2 rows for shell {shell} sum to {table_count}, "
                    f"but theta series gives {true_counts[shell]}"
                )

    def _validate_local_structures(self) -> None:
        for shell, structures in self.local_structures.items():
            for structure in structures:
                if structure.count != structure.leader.count:
                    raise ValueError(
                        f"local structure for shell={shell}, class={structure.class_index} "
                        f"counts {structure.count}, but Table 2 gives {structure.leader.count}"
                    )


def multiset_permutation_count(multiplicities: dict[int, int]) -> int:
    total = sum(multiplicities.values())
    out = factorial(total)
    for count in multiplicities.values():
        out //= factorial(count)
    return out


def extended_golay_codewords() -> tuple[tuple[int, ...], ...]:
    """Return G24 codewords in deterministic lexicographic order."""
    generator_exponents = (11, 9, 7, 6, 5, 1, 0)
    generator = sum(1 << exponent for exponent in generator_exponents)
    words = []
    for message in range(1 << 12):
        code23 = 0
        for bit in range(12):
            if (message >> bit) & 1:
                code23 ^= generator << bit
        word23 = tuple((code23 >> bit) & 1 for bit in range(23))
        parity = sum(word23) & 1
        words.append(word23 + (parity,))
    return tuple(sorted(words))


def build_local_structure(
    shell: int,
    class_index: int,
    leader: ClassLeader,
    golay_codewords: Sequence[tuple[int, ...]],
) -> ClassLocalStructure:
    if leader.parity == "even":
        target_weight = leader.mult_abs6 + leader.mult_abs2
        golay_indices = tuple(
            i for i, codeword in enumerate(golay_codewords) if sum(codeword) == target_weight
        )
        f0_multiplicities = {
            8: leader.mult_abs8,
            4: leader.mult_abs4,
            0: leader.mult_abs0,
        }
        f1_multiplicities = {
            6: leader.mult_abs6,
            2: leader.mult_abs2,
        }
        permutation_count = (
            multiset_permutation_count(f0_multiplicities)
            * multiset_permutation_count(f1_multiplicities)
        )
        sign_count = even_sign_count(leader)
    elif leader.parity == "odd":
        golay_indices = tuple(range(len(golay_codewords)))
        permutation_count = multiset_permutation_count(
            {
                5: leader.mult_abs5,
                3: leader.mult_abs3,
                1: leader.mult_abs1,
            }
        )
        sign_count = 1
    else:
        raise ValueError(f"unknown parity {leader.parity!r}")

    return ClassLocalStructure(
        shell=shell,
        class_index=class_index,
        leader=leader,
        golay_codeword_indices=golay_indices,
        sign_count=sign_count,
        permutation_count=permutation_count,
    )


def unrank_class_vector(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    if leader.parity == "even":
        return unrank_even_class_vector(leader, local)
    if leader.parity == "odd":
        return unrank_odd_class_vector(leader, local)
    raise ValueError(f"unknown parity {leader.parity!r}")


def unrank_even_class_vector(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    f0_positions = tuple(i for i, bit in enumerate(local.golay_codeword) if bit == 0)
    f1_positions = tuple(i for i, bit in enumerate(local.golay_codeword) if bit == 1)
    f0_counts = {8: leader.mult_abs8, 4: leader.mult_abs4, 0: leader.mult_abs0}
    f1_counts = {6: leader.mult_abs6, 2: leader.mult_abs2}
    f1_permutation_count = multiset_permutation_count(f1_counts)

    f1_rank = local.permutation_choice % f1_permutation_count
    f0_rank = local.permutation_choice // f1_permutation_count

    values = [0] * DIM
    for pos, magnitude in zip(f0_positions, unrank_multiset_sequence(f0_counts, f0_rank), strict=True):
        values[pos] = magnitude
    for pos, magnitude in zip(f1_positions, unrank_multiset_sequence(f1_counts, f1_rank), strict=True):
        values[pos] = magnitude
    return apply_even_signs(tuple(values), local.sign_choice)


def unrank_odd_class_vector(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    counts = {5: leader.mult_abs5, 3: leader.mult_abs3, 1: leader.mult_abs1}
    magnitudes = unrank_multiset_sequence(counts, local.permutation_choice)

    values = []
    for magnitude, bit in zip(magnitudes, local.golay_codeword, strict=True):
        target_mod4 = 3 if bit else 1
        sign = 1 if magnitude % 4 == target_mod4 else -1
        values.append(sign * magnitude)
    return tuple(values)


def unrank_multiset_sequence(multiplicities: dict[int, int], rank: int) -> tuple[int, ...]:
    counts = {value: count for value, count in multiplicities.items() if count}
    total = multiset_permutation_count(counts)
    if rank < 0 or rank >= total:
        raise IndexError(rank)

    values = tuple(sorted(counts, reverse=True))
    out = []
    remaining = sum(counts.values())
    while remaining:
        for value in values:
            count = counts.get(value, 0)
            if count == 0:
                continue
            counts[value] = count - 1
            branch_count = multiset_permutation_count(counts)
            if rank < branch_count:
                out.append(value)
                remaining -= 1
                break
            rank -= branch_count
            counts[value] = count
        else:
            raise RuntimeError("failed to unrank multiset permutation")
    return tuple(out)


@lru_cache(maxsize=None)
def _even_sign_count_from_groups(f0_values: tuple[int, ...], f1_values: tuple[int, ...]) -> int:
    unsigned = f0_values + f1_values
    signable = tuple(i for i, value in enumerate(unsigned) if value != 0)
    count = 0
    for mask in range(1 << len(signable)):
        total = 0
        for bit, pos in enumerate(signable):
            sign = -1 if (mask >> bit) & 1 else 1
            total += sign * unsigned[pos]
        if total % 8 == 0:
            count += 1
    return count


def even_sign_count(leader: ClassLeader) -> int:
    f0_values = (8,) * leader.mult_abs8 + (4,) * leader.mult_abs4 + (0,) * leader.mult_abs0
    f1_values = (6,) * leader.mult_abs6 + (2,) * leader.mult_abs2
    return _even_sign_count_from_groups(f0_values, f1_values)


def apply_even_signs(unsigned: tuple[int, ...], sign_rank: int) -> tuple[int, ...]:
    signable = tuple(i for i, value in enumerate(unsigned) if value != 0)
    for mask in range(1 << len(signable)):
        values = list(unsigned)
        for bit, pos in enumerate(signable):
            if (mask >> bit) & 1:
                values[pos] = -values[pos]
        if sum(values) % 8 != 0:
            continue
        if sign_rank == 0:
            return tuple(values)
        sign_rank -= 1
    raise IndexError(sign_rank)


def _class_sort_key(row: ClassLeader) -> tuple[tuple[int, ...], int, int, int]:
    parity_order = 0 if row.parity == "even" else 1
    table_position = TABLE2_CLASS_LEADERS.index(row)
    return (tuple(row.leader().tolist()), parity_order, row.count, table_position)


def demo(max_shell: int = 4, indices: str = "0,1103,1104,196559,196560") -> None:
    index = TrueLeechClassLeaderIndex(max_shell=max_shell)
    print(f"available shells: 2..{max_shell}")
    print(f"total class-indexed vectors: {index.total_count}")
    for shell in range(2, max_shell + 1):
        rows = index.classes_by_shell[shell]
        print(f"shell m={shell}: {len(rows)} classes, count={sum(row.count for row in rows)}")
        for class_idx, row in enumerate(rows):
            print(
                f"  class {class_idx}: parity={row.parity}, count={row.count}, "
                f"leader={row.leader().tolist()}"
            )

    print("rank/unrank examples:")
    for value in [int(part) for part in indices.split(",") if part.strip()]:
        ranked = index.unrank_class(value)
        reranked = index.rank_class(ranked.shell, ranked.class_index, ranked.class_local_index)
        print(
            f"  index={value} -> shell={ranked.shell}, class={ranked.class_index}, "
            f"class_local={ranked.class_local_index}, leader={ranked.leader.leader().tolist()}, "
            f"rerank={reranked}"
        )


def _argparse_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="True Leech Table-2 class leader rank/unrank")
    parser.add_argument("--max_shell", type=int, default=4)
    parser.add_argument("--indices", type=str, default="0,1103,1104,196559,196560")
    args = parser.parse_args()
    demo(max_shell=args.max_shell, indices=args.indices)


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        _argparse_main()
    else:
        fire.Fire(demo)
