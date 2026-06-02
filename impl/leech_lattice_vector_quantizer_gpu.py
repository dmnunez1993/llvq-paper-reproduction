from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, comb, factorial, log2, sqrt
from pathlib import Path
import pickle
from random import Random
import sys
from time import perf_counter
from typing import Iterable, Sequence

import torch
import triton
import triton.language as tl

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from impl.llvq_cuda_dequantize import dequantize_lattice_cuda
except Exception:  # pragma: no cover - optional CUDA driver/NVRTC path
    dequantize_lattice_cuda = None

try:
    from impl.llvq_cuda_quantize import quantize_lattice_cuda
except Exception:  # pragma: no cover - optional CUDA driver/NVRTC path
    quantize_lattice_cuda = None


DIM = 24
LATTICE_SCALE = 1.0 / sqrt(8.0)
CLASS_LEADER_CACHE_VERSION = 1


@triton.jit
def _comb_count(counts, comb_ptr, max_vals: tl.constexpr):
    remaining = tl.sum(counts, axis=0)
    out = tl.full((), 1, dtype=tl.int64)
    for i in tl.static_range(0, max_vals):
        chosen = tl.load(counts + i)
        out *= tl.load(comb_ptr + remaining * 25 + chosen)
        remaining -= chosen
    return out


@triton.jit
def _prefix_sign_count(mags, prefix_len, target, dim: tl.constexpr):
    residues = tl.arange(0, 8)
    dp = tl.where(residues == 0, 1, 0).to(tl.int64)
    for pos in tl.static_range(0, dim):
        active = pos < prefix_len
        mag = tl.load(mags + pos) & 7
        plus_idx = (residues - mag) & 7
        minus_idx = (residues + mag) & 7
        next_dp = tl.load(dp + plus_idx) + tl.load(dp + minus_idx)
        dp = tl.where(active, next_dp, dp)
    return tl.load(dp + target)


@triton.jit
def _dequantize_lattice_kernel(
    local_ptr,
    class_id_ptr,
    out_ptr,
    parity_ptr,
    perm_count_ptr,
    sign_count_ptr,
    f1_perm_ptr,
    golay_start_ptr,
    golay_indices_ptr,
    golay_bits_ptr,
    f0_counts_ptr,
    f1_counts_ptr,
    odd_counts_ptr,
    f0_values_ptr,
    f1_values_ptr,
    odd_values_ptr,
    comb_ptr,
    n: tl.constexpr,
    dim: tl.constexpr,
    max_vals: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n:
        return

    local = tl.load(local_ptr + row).to(tl.int64)
    class_id = tl.load(class_id_ptr + row).to(tl.int64)
    parity = tl.load(parity_ptr + class_id).to(tl.int32)
    perm_count = tl.load(perm_count_ptr + class_id).to(tl.int64)
    sign_count = tl.load(sign_count_ptr + class_id).to(tl.int64)

    permutation_choice = local % perm_count
    cursor = local // perm_count
    sign_choice = cursor % sign_count
    golay_choice = cursor // sign_count
    golay_start = tl.load(golay_start_ptr + class_id).to(tl.int64)
    golay_index = tl.load(golay_indices_ptr + golay_start + golay_choice).to(tl.int64)
    golay_bits = tl.load(golay_bits_ptr + golay_index).to(tl.int32)

    # Local scratch for unsigned/codeword magnitudes.
    mags = tl.full((DIM,), 0, dtype=tl.int64)
    compact = tl.full((DIM,), 0, dtype=tl.int64)
    compact_pos = tl.full((DIM,), 0, dtype=tl.int64)

    if parity == 0:
        f1_perm = tl.load(f1_perm_ptr + class_id).to(tl.int64)
        f1_rank = permutation_choice % f1_perm
        f0_rank = permutation_choice // f1_perm

        # Unrank f0 sequence onto Golay-zero positions.
        f0_counts = tl.load(f0_counts_ptr + class_id * max_vals + tl.arange(0, max_vals)).to(tl.int64)
        f0_width = tl.sum(f0_counts, axis=0)
        for seq_pos in tl.static_range(0, dim):
            active_seq = seq_pos < f0_width
            chosen_value = tl.full((), 0, dtype=tl.int64)
            chosen_vi = tl.full((), 0, dtype=tl.int64)
            chosen = tl.full((), False, dtype=tl.int1)
            for vi in tl.static_range(0, max_vals):
                count = tl.load(f0_counts + vi)
                value = tl.load(f0_values_ptr + class_id * max_vals + vi).to(tl.int64)
                eligible = active_seq & (~chosen) & (count > 0)
                tl.store(f0_counts + vi, count - tl.where(eligible, 1, 0))
                branch = _comb_count(f0_counts, comb_ptr, max_vals)
                take = eligible & (f0_rank < branch)
                f0_rank -= tl.where(eligible & (~take), branch, 0)
                tl.store(f0_counts + vi, count - tl.where(take, 1, 0))
                chosen_value = tl.where(take, value, chosen_value)
                chosen_vi = tl.where(take, vi, chosen_vi)
                chosen = chosen | take

            seen = tl.full((), 0, dtype=tl.int64)
            for d in tl.static_range(0, dim):
                bit = (golay_bits >> d) & 1
                is_pos = bit == 0
                place = active_seq & is_pos & (seen == seq_pos)
                tl.store(mags + d, tl.where(place, chosen_value, tl.load(mags + d)))
                seen += tl.where(is_pos, 1, 0)

        # Unrank f1 sequence onto Golay-one positions.
        f1_counts = tl.load(f1_counts_ptr + class_id * max_vals + tl.arange(0, max_vals)).to(tl.int64)
        f1_width = tl.sum(f1_counts, axis=0)
        for seq_pos in tl.static_range(0, dim):
            active_seq = seq_pos < f1_width
            chosen_value = tl.full((), 0, dtype=tl.int64)
            chosen = tl.full((), False, dtype=tl.int1)
            for vi in tl.static_range(0, max_vals):
                count = tl.load(f1_counts + vi)
                value = tl.load(f1_values_ptr + class_id * max_vals + vi).to(tl.int64)
                eligible = active_seq & (~chosen) & (count > 0)
                tl.store(f1_counts + vi, count - tl.where(eligible, 1, 0))
                branch = _comb_count(f1_counts, comb_ptr, max_vals)
                take = eligible & (f1_rank < branch)
                f1_rank -= tl.where(eligible & (~take), branch, 0)
                tl.store(f1_counts + vi, count - tl.where(take, 1, 0))
                chosen_value = tl.where(take, value, chosen_value)
                chosen = chosen | take

            seen = tl.full((), 0, dtype=tl.int64)
            for d in tl.static_range(0, dim):
                bit = (golay_bits >> d) & 1
                is_pos = bit == 1
                place = active_seq & is_pos & (seen == seq_pos)
                tl.store(mags + d, tl.where(place, chosen_value, tl.load(mags + d)))
                seen += tl.where(is_pos, 1, 0)

        # Compact nonzeros in coordinate order.
        nz = tl.full((), 0, dtype=tl.int64)
        for d in tl.static_range(0, dim):
            mag = tl.load(mags + d)
            is_nz = mag != 0
            tl.store(compact + nz, tl.where(is_nz, mag, tl.load(compact + nz)))
            tl.store(compact_pos + nz, tl.where(is_nz, d, tl.load(compact_pos + nz)))
            nz += tl.where(is_nz, 1, 0)

        signed_compact = tl.full((DIM,), 0, dtype=tl.int64)
        partial_high = tl.full((), 0, dtype=tl.int64)
        rank = sign_choice
        for rev in tl.static_range(0, dim):
            pos = dim - 1 - rev
            active = pos < nz
            mag = tl.load(compact + pos)
            needed = (-(partial_high + mag)) & 7
            zero_count = _prefix_sign_count(compact, pos, needed, dim)
            choose_neg = active & (rank >= zero_count)
            rank -= tl.where(choose_neg, zero_count, 0)
            signed = tl.where(choose_neg, -mag, mag)
            tl.store(signed_compact + pos, tl.where(active, signed, 0))
            partial_high += tl.where(active, signed, 0)

        for d in tl.static_range(0, dim):
            tl.store(out_ptr + row * dim + d, 0)
        for pos in tl.static_range(0, dim):
            active = pos < nz
            coord = tl.load(compact_pos + pos)
            val = tl.load(signed_compact + pos)
            tl.store(out_ptr + row * dim + coord, tl.where(active, val, 0))
    else:
        odd_counts = tl.load(odd_counts_ptr + class_id * max_vals + tl.arange(0, max_vals)).to(tl.int64)
        odd_rank = permutation_choice
        for seq_pos in tl.static_range(0, dim):
            chosen_value = tl.full((), 0, dtype=tl.int64)
            chosen = tl.full((), False, dtype=tl.int1)
            for vi in tl.static_range(0, max_vals):
                count = tl.load(odd_counts + vi)
                value = tl.load(odd_values_ptr + class_id * max_vals + vi).to(tl.int64)
                eligible = (~chosen) & (count > 0)
                tl.store(odd_counts + vi, count - tl.where(eligible, 1, 0))
                branch = _comb_count(odd_counts, comb_ptr, max_vals)
                take = eligible & (odd_rank < branch)
                odd_rank -= tl.where(eligible & (~take), branch, 0)
                tl.store(odd_counts + vi, count - tl.where(take, 1, 0))
                chosen_value = tl.where(take, value, chosen_value)
                chosen = chosen | take
            bit = (golay_bits >> seq_pos) & 1
            target = tl.where(bit == 1, 3, 1)
            sign = tl.where((chosen_value & 3) == target, 1, -1)
            tl.store(out_ptr + row * dim + seq_pos, sign * chosen_value)


@triton.jit
def _vec_get(vec, idx, width: tl.constexpr):
    offs = tl.arange(0, width)
    return tl.sum(tl.where(offs == idx, vec, 0), axis=0)


@triton.jit
def _vec_set(vec, idx, value, width: tl.constexpr):
    offs = tl.arange(0, width)
    return tl.where(offs == idx, value, vec)


@triton.jit
def _comb_count_vec(counts, comb_ptr, comb_stride: tl.constexpr, max_vals: tl.constexpr):
    offs = tl.arange(0, max_vals)
    remaining = tl.sum(counts, axis=0)
    out = tl.full((), 1, dtype=tl.int64)
    for i in tl.static_range(0, max_vals):
        chosen = _vec_get(counts, i, max_vals)
        out *= tl.load(comb_ptr + remaining * comb_stride + chosen)
        remaining -= chosen
    return out


@triton.jit
def _unrank_multiset_vec(
    rank,
    counts_ptr,
    values_ptr,
    class_id,
    comb_ptr,
    dim: tl.constexpr,
    block_dim: tl.constexpr,
    comb_stride: tl.constexpr,
    max_vals: tl.constexpr,
):
    voffs = tl.arange(0, max_vals)
    counts = tl.load(counts_ptr + class_id * max_vals + voffs).to(tl.int64)
    values = tl.load(values_ptr + class_id * max_vals + voffs).to(tl.int64)
    width = tl.sum(counts, axis=0)
    seq = tl.full((block_dim,), 0, dtype=tl.int64)

    for pos in tl.static_range(0, dim):
        active_pos = pos < width
        chosen = tl.full((), False, dtype=tl.int1)
        chosen_value = tl.full((), 0, dtype=tl.int64)
        for vi in tl.static_range(0, max_vals):
            count = _vec_get(counts, vi, max_vals)
            eligible = active_pos & (~chosen) & (count > 0)
            trial = _vec_set(counts, vi, count - 1, max_vals)
            branch = _comb_count_vec(trial, comb_ptr, comb_stride, max_vals)
            take = eligible & (rank < branch)
            rank -= tl.where(eligible & (~take), branch, 0)
            counts = tl.where(take, trial, counts)
            chosen_value = tl.where(take, _vec_get(values, vi, max_vals), chosen_value)
            chosen = chosen | take
        seq = _vec_set(seq, pos, chosen_value, block_dim)
    return seq


@triton.jit
def _select8(idx, v0, v1, v2, v3, v4, v5, v6, v7):
    out = tl.where(idx == 0, v0, v1)
    out = tl.where(idx == 2, v2, out)
    out = tl.where(idx == 3, v3, out)
    out = tl.where(idx == 4, v4, out)
    out = tl.where(idx == 5, v5, out)
    out = tl.where(idx == 6, v6, out)
    out = tl.where(idx == 7, v7, out)
    return out


@triton.jit
def _prefix_sign_count_vec(compact, prefix_len, target, dim: tl.constexpr, block_dim: tl.constexpr):
    dp0 = tl.full((), 1, dtype=tl.int64)
    dp1 = tl.full((), 0, dtype=tl.int64)
    dp2 = tl.full((), 0, dtype=tl.int64)
    dp3 = tl.full((), 0, dtype=tl.int64)
    dp4 = tl.full((), 0, dtype=tl.int64)
    dp5 = tl.full((), 0, dtype=tl.int64)
    dp6 = tl.full((), 0, dtype=tl.int64)
    dp7 = tl.full((), 0, dtype=tl.int64)
    for pos in tl.static_range(0, dim):
        active = pos < prefix_len
        mag = _vec_get(compact, pos, block_dim) & 7
        n0 = _select8((0 - mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7) + _select8((0 + mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
        n1 = _select8((1 - mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7) + _select8((1 + mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
        n2 = _select8((2 - mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7) + _select8((2 + mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
        n3 = _select8((3 - mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7) + _select8((3 + mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
        n4 = _select8((4 - mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7) + _select8((4 + mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
        n5 = _select8((5 - mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7) + _select8((5 + mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
        n6 = _select8((6 - mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7) + _select8((6 + mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
        n7 = _select8((7 - mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7) + _select8((7 + mag) & 7, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)
        dp0 = tl.where(active, n0, dp0)
        dp1 = tl.where(active, n1, dp1)
        dp2 = tl.where(active, n2, dp2)
        dp3 = tl.where(active, n3, dp3)
        dp4 = tl.where(active, n4, dp4)
        dp5 = tl.where(active, n5, dp5)
        dp6 = tl.where(active, n6, dp6)
        dp7 = tl.where(active, n7, dp7)
    return _select8(target, dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7)


@triton.jit
def _even_lower_sign_count(compact, prefix_len, target, dim: tl.constexpr, block_dim: tl.constexpr):
    fixed = tl.full((), 0, dtype=tl.int64)
    f0 = tl.full((), 0, dtype=tl.int64)
    f1 = tl.full((), 0, dtype=tl.int64)
    for pos in tl.static_range(0, dim):
        active = pos < prefix_len
        mag = _vec_get(compact, pos, block_dim)
        nz = active & (mag != 0)
        fixed += tl.where(nz, mag & 7, 0)
        is_f1 = (mag & 3) == 2
        f1 += tl.where(nz & is_f1, 1, 0)
        f0 += tl.where(nz & (~is_f1), 1, 0)

    delta = (target - fixed) & 7
    valid = (delta == 0) | (delta == 4)
    exponent = f0 + tl.maximum(f1 - 1, 0)
    count = tl.full((), 1, dtype=tl.int64) << exponent
    no_f1_count = tl.where(delta == 0, tl.full((), 1, dtype=tl.int64) << f0, 0)
    return tl.where(valid, tl.where(f1 > 0, count, no_f1_count), 0)


@triton.jit
def _dequantize_lattice_kernel_v2(
    local_ptr,
    class_id_ptr,
    out_ptr,
    parity_ptr,
    perm_count_ptr,
    sign_count_ptr,
    f1_perm_ptr,
    golay_start_ptr,
    golay_indices_ptr,
    golay_bits_ptr,
    f0_counts_ptr,
    f1_counts_ptr,
    odd_counts_ptr,
    f0_values_ptr,
    f1_values_ptr,
    odd_values_ptr,
    comb_ptr,
    n: tl.constexpr,
    dim: tl.constexpr,
    block_dim: tl.constexpr,
    comb_stride: tl.constexpr,
    max_vals: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n:
        return

    local = tl.load(local_ptr + row).to(tl.int64)
    class_id = tl.load(class_id_ptr + row).to(tl.int64)
    parity = tl.load(parity_ptr + class_id).to(tl.int32)
    perm_count = tl.load(perm_count_ptr + class_id).to(tl.int64)
    sign_count = tl.load(sign_count_ptr + class_id).to(tl.int64)
    permutation_choice = local % perm_count
    cursor = local // perm_count
    sign_choice = cursor % sign_count
    golay_choice = cursor // sign_count
    golay_start = tl.load(golay_start_ptr + class_id).to(tl.int64)
    golay_index = tl.load(golay_indices_ptr + golay_start + golay_choice).to(tl.int64)
    golay_bits = tl.load(golay_bits_ptr + golay_index).to(tl.int32)

    outv = tl.full((block_dim,), 0, dtype=tl.int64)
    if parity == 0:
        f1_perm = tl.load(f1_perm_ptr + class_id).to(tl.int64)
        f1_rank = permutation_choice % f1_perm
        f0_rank = permutation_choice // f1_perm
        f0_seq = _unrank_multiset_vec(f0_rank, f0_counts_ptr, f0_values_ptr, class_id, comb_ptr, dim, block_dim, comb_stride, max_vals)
        f1_seq = _unrank_multiset_vec(f1_rank, f1_counts_ptr, f1_values_ptr, class_id, comb_ptr, dim, block_dim, comb_stride, max_vals)

        seen0 = tl.full((), 0, dtype=tl.int64)
        seen1 = tl.full((), 0, dtype=tl.int64)
        unsigned = tl.full((block_dim,), 0, dtype=tl.int64)
        compact = tl.full((block_dim,), 0, dtype=tl.int64)
        compact_pos = tl.full((block_dim,), 0, dtype=tl.int64)
        nz = tl.full((), 0, dtype=tl.int64)
        for d in tl.static_range(0, dim):
            bit = (golay_bits >> d) & 1
            mag0 = _vec_get(f0_seq, seen0, block_dim)
            mag1 = _vec_get(f1_seq, seen1, block_dim)
            mag = tl.where(bit == 0, mag0, mag1)
            seen0 += tl.where(bit == 0, 1, 0)
            seen1 += tl.where(bit == 1, 1, 0)
            unsigned = _vec_set(unsigned, d, mag, block_dim)
            is_nz = mag != 0
            compact = tl.where(is_nz, _vec_set(compact, nz, mag, block_dim), compact)
            compact_pos = tl.where(is_nz, _vec_set(compact_pos, nz, d, block_dim), compact_pos)
            nz += tl.where(is_nz, 1, 0)

        signed_compact = tl.full((block_dim,), 0, dtype=tl.int64)
        partial_high = tl.full((), 0, dtype=tl.int64)
        rank = sign_choice
        for rev in tl.static_range(0, dim):
            pos = dim - 1 - rev
            active = pos < nz
            mag = _vec_get(compact, pos, block_dim)
            needed = (-(partial_high + mag)) & 7
            zero_count = _prefix_sign_count_vec(compact, pos, needed, dim, block_dim)
            choose_neg = active & (rank >= zero_count)
            rank -= tl.where(choose_neg, zero_count, 0)
            signed = tl.where(choose_neg, -mag, mag)
            signed_compact = tl.where(active, _vec_set(signed_compact, pos, signed, block_dim), signed_compact)
            partial_high += tl.where(active, signed, 0)

        for pos in tl.static_range(0, dim):
            active = pos < nz
            coord = _vec_get(compact_pos, pos, block_dim)
            val = _vec_get(signed_compact, pos, block_dim)
            outv = tl.where(active, _vec_set(outv, coord, val, block_dim), outv)
    else:
        odd_seq = _unrank_multiset_vec(permutation_choice, odd_counts_ptr, odd_values_ptr, class_id, comb_ptr, dim, block_dim, comb_stride, max_vals)
        for d in tl.static_range(0, dim):
            mag = _vec_get(odd_seq, d, block_dim)
            bit = (golay_bits >> d) & 1
            target = tl.where(bit == 1, 3, 1)
            sign = tl.where((mag & 3) == target, 1, -1)
            outv = _vec_set(outv, d, sign * mag, block_dim)

    offs = tl.arange(0, block_dim)
    tl.store(out_ptr + row * dim + offs, outv, mask=offs < dim)


@triton.jit
def _dequantize_lattice_odd_kernel(
    local_ptr,
    class_id_ptr,
    out_ptr,
    parity_ptr,
    perm_count_ptr,
    golay_start_ptr,
    golay_indices_ptr,
    golay_bits_ptr,
    odd_counts_ptr,
    odd_values_ptr,
    comb_ptr,
    n: tl.constexpr,
    dim: tl.constexpr,
    block_dim: tl.constexpr,
    comb_stride: tl.constexpr,
    max_vals: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n:
        return

    local = tl.load(local_ptr + row).to(tl.int64)
    class_id = tl.load(class_id_ptr + row).to(tl.int64)
    parity = tl.load(parity_ptr + class_id).to(tl.int32)
    if parity != 1:
        return

    perm_count = tl.load(perm_count_ptr + class_id).to(tl.int64)
    permutation_choice = local % perm_count
    golay_choice = local // perm_count
    golay_start = tl.load(golay_start_ptr + class_id).to(tl.int64)
    golay_index = tl.load(golay_indices_ptr + golay_start + golay_choice).to(tl.int64)
    golay_bits = tl.load(golay_bits_ptr + golay_index).to(tl.int32)
    odd_seq = _unrank_multiset_vec(
        permutation_choice,
        odd_counts_ptr,
        odd_values_ptr,
        class_id,
        comb_ptr,
        dim,
        block_dim,
        comb_stride,
        max_vals,
    )

    outv = tl.full((block_dim,), 0, dtype=tl.int64)
    for d in tl.static_range(0, dim):
        mag = _vec_get(odd_seq, d, block_dim)
        bit = (golay_bits >> d) & 1
        target = tl.where(bit == 1, 3, 1)
        sign = tl.where((mag & 3) == target, 1, -1)
        outv = _vec_set(outv, d, sign * mag, block_dim)

    offs = tl.arange(0, block_dim)
    tl.store(out_ptr + row * dim + offs, outv, mask=offs < dim)


@triton.jit
def _dequantize_lattice_even_kernel(
    local_ptr,
    class_id_ptr,
    out_ptr,
    parity_ptr,
    perm_count_ptr,
    sign_count_ptr,
    f1_perm_ptr,
    golay_start_ptr,
    golay_indices_ptr,
    golay_bits_ptr,
    f0_counts_ptr,
    f1_counts_ptr,
    f0_values_ptr,
    f1_values_ptr,
    comb_ptr,
    n: tl.constexpr,
    dim: tl.constexpr,
    block_dim: tl.constexpr,
    comb_stride: tl.constexpr,
    max_vals: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= n:
        return

    local = tl.load(local_ptr + row).to(tl.int64)
    class_id = tl.load(class_id_ptr + row).to(tl.int64)
    parity = tl.load(parity_ptr + class_id).to(tl.int32)
    if parity != 0:
        return

    perm_count = tl.load(perm_count_ptr + class_id).to(tl.int64)
    sign_count = tl.load(sign_count_ptr + class_id).to(tl.int64)
    permutation_choice = local % perm_count
    cursor = local // perm_count
    sign_choice = cursor % sign_count
    golay_choice = cursor // sign_count
    golay_start = tl.load(golay_start_ptr + class_id).to(tl.int64)
    golay_index = tl.load(golay_indices_ptr + golay_start + golay_choice).to(tl.int64)
    golay_bits = tl.load(golay_bits_ptr + golay_index).to(tl.int32)

    f1_perm = tl.load(f1_perm_ptr + class_id).to(tl.int64)
    f1_rank = permutation_choice % f1_perm
    f0_rank = permutation_choice // f1_perm
    f0_seq = _unrank_multiset_vec(
        f0_rank,
        f0_counts_ptr,
        f0_values_ptr,
        class_id,
        comb_ptr,
        dim,
        block_dim,
        comb_stride,
        max_vals,
    )
    f1_seq = _unrank_multiset_vec(
        f1_rank,
        f1_counts_ptr,
        f1_values_ptr,
        class_id,
        comb_ptr,
        dim,
        block_dim,
        comb_stride,
        max_vals,
    )

    seen0 = tl.full((), 0, dtype=tl.int64)
    seen1 = tl.full((), 0, dtype=tl.int64)
    compact = tl.full((block_dim,), 0, dtype=tl.int64)
    compact_pos = tl.full((block_dim,), 0, dtype=tl.int64)
    nz = tl.full((), 0, dtype=tl.int64)
    for d in tl.static_range(0, dim):
        bit = (golay_bits >> d) & 1
        mag0 = _vec_get(f0_seq, seen0, block_dim)
        mag1 = _vec_get(f1_seq, seen1, block_dim)
        mag = tl.where(bit == 0, mag0, mag1)
        seen0 += tl.where(bit == 0, 1, 0)
        seen1 += tl.where(bit == 1, 1, 0)
        is_nz = mag != 0
        compact = tl.where(is_nz, _vec_set(compact, nz, mag, block_dim), compact)
        compact_pos = tl.where(is_nz, _vec_set(compact_pos, nz, d, block_dim), compact_pos)
        nz += tl.where(is_nz, 1, 0)

    signed_compact = tl.full((block_dim,), 0, dtype=tl.int64)
    partial_high = tl.full((), 0, dtype=tl.int64)
    rank = sign_choice
    for rev in tl.static_range(0, dim):
        pos = dim - 1 - rev
        active = pos < nz
        mag = _vec_get(compact, pos, block_dim)
        needed = (-(partial_high + mag)) & 7
        zero_count = _even_lower_sign_count(compact, pos, needed, dim, block_dim)
        choose_neg = active & (rank >= zero_count)
        rank -= tl.where(choose_neg, zero_count, 0)
        signed = tl.where(choose_neg, -mag, mag)
        signed_compact = tl.where(active, _vec_set(signed_compact, pos, signed, block_dim), signed_compact)
        partial_high += tl.where(active, signed, 0)

    outv = tl.full((block_dim,), 0, dtype=tl.int64)
    for pos in tl.static_range(0, dim):
        active = pos < nz
        coord = _vec_get(compact_pos, pos, block_dim)
        val = _vec_get(signed_compact, pos, block_dim)
        outv = tl.where(active, _vec_set(outv, coord, val, block_dim), outv)

    offs = tl.arange(0, block_dim)
    tl.store(out_ptr + row * dim + offs, outv, mask=offs < dim)


@triton.jit
def _score_odd_scalar_kernel(
    x_ptr,
    bits_mask_ptr,
    signed_leaders_ptr,
    shell_ptr,
    out_score_ptr,
    n_subclasses: tl.constexpr,
    dim: tl.constexpr,
):
    batch_id = tl.program_id(0)
    subclass_id = tl.program_id(1)

    selected = tl.full((), 0, dtype=tl.int32)
    projection = tl.full((), 0.0, dtype=tl.float32)
    for rank in tl.static_range(0, dim):
        best_val = tl.full((), float("inf"), dtype=tl.float32)
        best_bit = tl.full((), 0, dtype=tl.int32)
        for d in tl.static_range(0, dim):
            bit_mask = tl.full((), 1 << d, dtype=tl.int32)
            free = (selected & bit_mask) == 0
            golay_bit = (tl.load(bits_mask_ptr + subclass_id).to(tl.int32) >> d) & 1
            flip = tl.where(golay_bit == 1, -1.0, 1.0)
            x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
            x_prime = x_val * flip
            better = free & (x_prime < best_val)
            best_val = tl.where(better, x_prime, best_val)
            best_bit = tl.where(better, bit_mask, best_bit)
        selected = selected | best_bit
        leader_val = tl.load(signed_leaders_ptr + subclass_id * dim + rank).to(tl.float32)
        projection += best_val * leader_val

    shell = tl.load(shell_ptr + subclass_id).to(tl.float32)
    score = 0.35355339059327373 * projection - shell
    tl.store(out_score_ptr + batch_id * n_subclasses + subclass_id, score)


@triton.jit
def _score_even_scalar_kernel(
    x_ptr,
    f0_mask_ptr,
    f1_mask_ptr,
    f0_mags_ptr,
    f1_mags_ptr,
    f0_len_ptr,
    f1_len_ptr,
    req_ptr,
    shell_ptr,
    out_score_ptr,
    n_subclasses: tl.constexpr,
    dim: tl.constexpr,
):
    batch_id = tl.program_id(0)
    subclass_id = tl.program_id(1)

    projection = tl.full((), 0.0, dtype=tl.float32)

    selected0 = tl.full((), 0, dtype=tl.int32)
    f0_mask = tl.load(f0_mask_ptr + subclass_id).to(tl.int32)
    f0_len = tl.load(f0_len_ptr + subclass_id).to(tl.int32)
    for pick in tl.static_range(0, dim):
        active = pick < f0_len
        best_abs = tl.full((), float("inf"), dtype=tl.float32)
        best_bit = tl.full((), 0, dtype=tl.int32)
        for d in tl.static_range(0, dim):
            bit_mask = tl.full((), 1 << d, dtype=tl.int32)
            allowed = ((f0_mask & bit_mask) != 0) & ((selected0 & bit_mask) == 0)
            x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
            abs_val = tl.abs(x_val)
            better = active & allowed & (abs_val < best_abs)
            best_abs = tl.where(better, abs_val, best_abs)
            best_bit = tl.where(better, bit_mask, best_bit)
        selected0 = selected0 | best_bit
        mag = tl.load(f0_mags_ptr + subclass_id * dim + pick).to(tl.float32)
        projection += tl.where(active, mag * best_abs, 0.0)

    selected1 = tl.full((), 0, dtype=tl.int32)
    f1_mask = tl.load(f1_mask_ptr + subclass_id).to(tl.int32)
    f1_len = tl.load(f1_len_ptr + subclass_id).to(tl.int32)
    negative_parity = tl.full((), 0, dtype=tl.int32)
    first_loss = tl.full((), 0.0, dtype=tl.float32)
    for pick in tl.static_range(0, dim):
        active = pick < f1_len
        best_abs = tl.full((), float("inf"), dtype=tl.float32)
        best_bit = tl.full((), 0, dtype=tl.int32)
        best_neg = tl.full((), 0, dtype=tl.int32)
        for d in tl.static_range(0, dim):
            bit_mask = tl.full((), 1 << d, dtype=tl.int32)
            allowed = ((f1_mask & bit_mask) != 0) & ((selected1 & bit_mask) == 0)
            x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
            abs_val = tl.abs(x_val)
            better = active & allowed & (abs_val < best_abs)
            best_abs = tl.where(better, abs_val, best_abs)
            best_bit = tl.where(better, bit_mask, best_bit)
            best_neg = tl.where(better, tl.where(x_val < 0.0, 1, 0), best_neg)
        selected1 = selected1 | best_bit
        mag = tl.load(f1_mags_ptr + subclass_id * dim + pick).to(tl.float32)
        projection += tl.where(active, mag * best_abs, 0.0)
        negative_parity = tl.where(active, (negative_parity + best_neg) & 1, negative_parity)
        first_loss = tl.where(active & (pick == 0), 2.0 * mag * best_abs, first_loss)

    required = tl.load(req_ptr + subclass_id).to(tl.int32)
    needs_flip = (f1_len > 0) & (negative_parity != required)
    projection = tl.where(needs_flip, projection - first_loss, projection)

    shell = tl.load(shell_ptr + subclass_id).to(tl.float32)
    score = 0.35355339059327373 * projection - shell
    tl.store(out_score_ptr + batch_id * n_subclasses + subclass_id, score)


@triton.jit
def _score_compact_pairs_kernel(
    x_ptr,
    golay_bits_ptr,
    golay_weight_ptr,
    class_parity_ptr,
    class_shell_ptr,
    class_even_weight_ptr,
    class_f0_mags_ptr,
    class_f1_mags_ptr,
    class_f0_len_ptr,
    class_f1_len_ptr,
    class_required_ptr,
    class_odd_leaders_ptr,
    out_score_ptr,
    pair_start: tl.constexpr,
    n_pairs: tl.constexpr,
    n_classes: tl.constexpr,
    n_golay: tl.constexpr,
    dim: tl.constexpr,
):
    batch_id = tl.program_id(0)
    pair_id = tl.program_id(1)
    global_pair = pair_start + pair_id
    class_id = global_pair // n_golay
    golay_index = global_pair - class_id * n_golay
    valid = (pair_id < n_pairs) & (class_id < n_classes)

    golay_bits = tl.load(golay_bits_ptr + golay_index, mask=valid, other=0).to(tl.int32)
    parity = tl.load(class_parity_ptr + class_id, mask=valid, other=0).to(tl.int32)
    shell = tl.load(class_shell_ptr + class_id, mask=valid, other=0.0).to(tl.float32)
    projection = tl.full((), 0.0, dtype=tl.float32)

    if parity == 0:
        weight = tl.load(golay_weight_ptr + golay_index, mask=valid, other=-1).to(tl.int32)
        even_weight = tl.load(class_even_weight_ptr + class_id, mask=valid, other=-2).to(tl.int32)
        valid = valid & (weight == even_weight)

        selected0 = tl.full((), 0, dtype=tl.int32)
        f0_mask = ((1 << 24) - 1) ^ golay_bits
        f0_len = tl.load(class_f0_len_ptr + class_id, mask=valid, other=0).to(tl.int32)
        for pick in tl.static_range(0, dim):
            active = valid & (pick < f0_len)
            best_abs = tl.full((), float("inf"), dtype=tl.float32)
            best_bit = tl.full((), 0, dtype=tl.int32)
            for d in tl.static_range(0, dim):
                bit_mask = tl.full((), 1 << d, dtype=tl.int32)
                allowed = ((f0_mask & bit_mask) != 0) & ((selected0 & bit_mask) == 0)
                x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
                abs_val = tl.abs(x_val)
                better = active & allowed & (abs_val < best_abs)
                best_abs = tl.where(better, abs_val, best_abs)
                best_bit = tl.where(better, bit_mask, best_bit)
            selected0 = selected0 | best_bit
            mag = tl.load(class_f0_mags_ptr + class_id * dim + pick, mask=active, other=0).to(tl.float32)
            projection += tl.where(active, mag * best_abs, 0.0)

        selected1 = tl.full((), 0, dtype=tl.int32)
        f1_mask = golay_bits
        f1_len = tl.load(class_f1_len_ptr + class_id, mask=valid, other=0).to(tl.int32)
        negative_parity = tl.full((), 0, dtype=tl.int32)
        first_loss = tl.full((), 0.0, dtype=tl.float32)
        for pick in tl.static_range(0, dim):
            active = valid & (pick < f1_len)
            best_abs = tl.full((), float("inf"), dtype=tl.float32)
            best_bit = tl.full((), 0, dtype=tl.int32)
            best_neg = tl.full((), 0, dtype=tl.int32)
            for d in tl.static_range(0, dim):
                bit_mask = tl.full((), 1 << d, dtype=tl.int32)
                allowed = ((f1_mask & bit_mask) != 0) & ((selected1 & bit_mask) == 0)
                x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
                abs_val = tl.abs(x_val)
                better = active & allowed & (abs_val < best_abs)
                best_abs = tl.where(better, abs_val, best_abs)
                best_bit = tl.where(better, bit_mask, best_bit)
                best_neg = tl.where(better, tl.where(x_val < 0.0, 1, 0), best_neg)
            selected1 = selected1 | best_bit
            mag = tl.load(class_f1_mags_ptr + class_id * dim + pick, mask=active, other=0).to(tl.float32)
            projection += tl.where(active, mag * best_abs, 0.0)
            negative_parity = tl.where(active, (negative_parity + best_neg) & 1, negative_parity)
            first_loss = tl.where(active & (pick == 0), 2.0 * mag * best_abs, first_loss)

        required = tl.load(class_required_ptr + class_id, mask=valid, other=0).to(tl.int32)
        needs_flip = valid & (f1_len > 0) & (negative_parity != required)
        projection = tl.where(needs_flip, projection - first_loss, projection)
    else:
        selected = tl.full((), 0, dtype=tl.int32)
        for rank in tl.static_range(0, dim):
            best_val = tl.full((), float("inf"), dtype=tl.float32)
            best_bit = tl.full((), 0, dtype=tl.int32)
            for d in tl.static_range(0, dim):
                bit_mask = tl.full((), 1 << d, dtype=tl.int32)
                free = (selected & bit_mask) == 0
                golay_bit = (golay_bits >> d) & 1
                flip = tl.where(golay_bit == 1, -1.0, 1.0)
                x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
                x_prime = x_val * flip
                better = valid & free & (x_prime < best_val)
                best_val = tl.where(better, x_prime, best_val)
                best_bit = tl.where(better, bit_mask, best_bit)
            selected = selected | best_bit
            leader_val = tl.load(class_odd_leaders_ptr + class_id * dim + rank, mask=valid, other=0).to(tl.float32)
            projection += best_val * leader_val

    score = 0.35355339059327373 * projection - shell
    score = tl.where(valid, score, float("-inf"))
    tl.store(out_score_ptr + batch_id * n_pairs + pair_id, score)


@triton.jit
def _score_odd_subclasses_kernel(
    x_ptr,
    bits_ptr,
    signed_leaders_ptr,
    shell_ptr,
    out_score_ptr,
    out_pos_ptr,
    n_subclasses: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offsets < n_subclasses

    selected = tl.zeros((BLOCK_N,), dtype=tl.int32)
    projection = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for rank in tl.static_range(0, dim):
        best_val = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best_bit = tl.zeros((BLOCK_N,), dtype=tl.int32)
        for d in tl.static_range(0, dim):
            bit_mask = tl.full((BLOCK_N,), 1 << d, dtype=tl.int32)
            free = (selected & bit_mask) == 0
            golay_bit = tl.load(bits_ptr + offsets * dim + d, mask=valid, other=0).to(tl.int32)
            flip = tl.where(golay_bit == 1, -1.0, 1.0)
            x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
            x_prime = x_val * flip
            better = valid & free & (x_prime < best_val)
            best_val = tl.where(better, x_prime, best_val)
            best_bit = tl.where(better, bit_mask, best_bit)
        selected = selected | best_bit
        leader_val = tl.load(signed_leaders_ptr + offsets * dim + rank, mask=valid, other=0).to(tl.float32)
        projection += best_val * leader_val

    shell = tl.load(shell_ptr + offsets, mask=valid, other=0).to(tl.float32)
    score = 0.35355339059327373 * projection - shell
    score = tl.where(valid, score, float("-inf"))
    best_score = tl.max(score, axis=0)
    best_pos = tl.argmax(score, axis=0)
    out = batch_id * tl.num_programs(1) + block_id
    tl.store(out_score_ptr + out, best_score)
    tl.store(out_pos_ptr + out, block_id * BLOCK_N + best_pos)


@triton.jit
def _score_even_subclasses_kernel(
    x_ptr,
    f0_mask_ptr,
    f1_mask_ptr,
    f0_mags_ptr,
    f1_mags_ptr,
    f0_len_ptr,
    f1_len_ptr,
    req_ptr,
    shell_ptr,
    out_score_ptr,
    out_pos_ptr,
    n_subclasses: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offsets < n_subclasses

    projection = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # f0 group: assign ascending magnitudes to ascending |x| positions where Golay bit is 0.
    selected0 = tl.zeros((BLOCK_N,), dtype=tl.int32)
    f0_mask = tl.load(f0_mask_ptr + offsets, mask=valid, other=0).to(tl.int32)
    f0_len = tl.load(f0_len_ptr + offsets, mask=valid, other=0).to(tl.int32)
    for pick in tl.static_range(0, dim):
        active = valid & (pick < f0_len)
        best_abs = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best_bit = tl.zeros((BLOCK_N,), dtype=tl.int32)
        for d in tl.static_range(0, dim):
            bit_mask = tl.full((BLOCK_N,), 1 << d, dtype=tl.int32)
            allowed = ((f0_mask & bit_mask) != 0) & ((selected0 & bit_mask) == 0)
            x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
            abs_val = tl.abs(x_val)
            better = active & allowed & (abs_val < best_abs)
            best_abs = tl.where(better, abs_val, best_abs)
            best_bit = tl.where(better, bit_mask, best_bit)
        selected0 = selected0 | best_bit
        mag = tl.load(f0_mags_ptr + offsets * dim + pick, mask=active, other=0).to(tl.float32)
        projection += tl.where(active, mag * best_abs, 0.0)

    # f1 group: same, plus the even-sign parity correction may flip the first f1 pick.
    selected1 = tl.zeros((BLOCK_N,), dtype=tl.int32)
    f1_mask = tl.load(f1_mask_ptr + offsets, mask=valid, other=0).to(tl.int32)
    f1_len = tl.load(f1_len_ptr + offsets, mask=valid, other=0).to(tl.int32)
    negative_parity = tl.zeros((BLOCK_N,), dtype=tl.int32)
    first_loss = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for pick in tl.static_range(0, dim):
        active = valid & (pick < f1_len)
        best_abs = tl.full((BLOCK_N,), float("inf"), dtype=tl.float32)
        best_bit = tl.zeros((BLOCK_N,), dtype=tl.int32)
        best_neg = tl.zeros((BLOCK_N,), dtype=tl.int32)
        for d in tl.static_range(0, dim):
            bit_mask = tl.full((BLOCK_N,), 1 << d, dtype=tl.int32)
            allowed = ((f1_mask & bit_mask) != 0) & ((selected1 & bit_mask) == 0)
            x_val = tl.load(x_ptr + batch_id * dim + d).to(tl.float32)
            abs_val = tl.abs(x_val)
            better = active & allowed & (abs_val < best_abs)
            best_abs = tl.where(better, abs_val, best_abs)
            best_bit = tl.where(better, bit_mask, best_bit)
            best_neg = tl.where(better, tl.where(x_val < 0.0, 1, 0), best_neg)
        selected1 = selected1 | best_bit
        mag = tl.load(f1_mags_ptr + offsets * dim + pick, mask=active, other=0).to(tl.float32)
        projection += tl.where(active, mag * best_abs, 0.0)
        negative_parity = tl.where(active, (negative_parity + best_neg) & 1, negative_parity)
        first_loss = tl.where(active & (pick == 0), 2.0 * mag * best_abs, first_loss)

    required = tl.load(req_ptr + offsets, mask=valid, other=0).to(tl.int32)
    needs_flip = valid & (f1_len > 0) & (negative_parity != required)
    projection = tl.where(needs_flip, projection - first_loss, projection)

    shell = tl.load(shell_ptr + offsets, mask=valid, other=0).to(tl.float32)
    score = 0.35355339059327373 * projection - shell
    score = tl.where(valid, score, float("-inf"))
    best_score = tl.max(score, axis=0)
    best_pos = tl.argmax(score, axis=0)
    out = batch_id * tl.num_programs(1) + block_id
    tl.store(out_score_ptr + out, best_score)
    tl.store(out_pos_ptr + out, block_id * BLOCK_N + best_pos)


@dataclass(frozen=True)
class ClassLeader:
    """Absolute-value leader for one Leech spherical-code class."""

    shell: int
    parity: str
    count: int
    mult_abs8: int
    mult_abs6: int
    mult_abs5: int
    mult_abs4: int
    mult_abs3: int
    mult_abs2: int
    mult_abs1: int
    mult_abs0: int
    extra_multiplicities: tuple[tuple[int, int], ...] = ()

    @property
    def multiplicities(self) -> dict[int, int]:
        out = {
            8: self.mult_abs8,
            6: self.mult_abs6,
            5: self.mult_abs5,
            4: self.mult_abs4,
            3: self.mult_abs3,
            2: self.mult_abs2,
            1: self.mult_abs1,
            0: self.mult_abs0,
        }
        for magnitude, count in self.extra_multiplicities:
            out[magnitude] = count
        return {magnitude: count for magnitude, count in out.items() if count}

    @property
    def leader(self) -> tuple[int, ...]:
        values: list[int] = []
        for magnitude, count in sorted(self.multiplicities.items(), reverse=True):
            values.extend([magnitude] * count)
        if len(values) != DIM:
            raise ValueError(f"leader has {len(values)} coordinates, expected {DIM}")
        return tuple(values)

    @property
    def even_weight(self) -> int:
        """Number of even leader coordinates congruent to 2 mod 4."""
        return sum(count for magnitude, count in self.multiplicities.items() if magnitude % 4 == 2)

    @property
    def nonzero_count(self) -> int:
        return DIM - self.mult_abs0


@dataclass(frozen=True)
class RankedCodeword:
    global_index: int
    shell: int
    shell_local_index: int
    class_index: int
    class_local_index: int
    class_offset: int
    leader: ClassLeader


@dataclass(frozen=True)
class LocalDecomposition:
    golay_choice: int
    golay_codeword: tuple[int, ...]
    sign_choice: int
    permutation_choice: int


@dataclass(frozen=True)
class ClassLocalStructure:
    shell: int
    class_index: int
    leader: ClassLeader
    golay_codeword_indices: tuple[int, ...]
    sign_count: int
    permutation_count: int

    @property
    def count(self) -> int:
        return len(self.golay_codeword_indices) * self.sign_count * self.permutation_count


# Reference leaders/counts from the Adoul-Barth small-shell leader table for m=2..4.
# These are not used to build the database; generation is done from the shell
# equation and class-construction rules, then compared against this table.
# Multiplicity columns are |8|, |6|, |5|, |4|, |3|, |2|, |1|, |0|.
TABLE_CLASS_LEADERS_REFERENCE: tuple[ClassLeader, ...] = (
    ClassLeader(2, "even", 1104, 0, 0, 0, 2, 0, 0, 0, 22),
    ClassLeader(2, "even", 97152, 0, 0, 0, 0, 0, 8, 0, 16),
    ClassLeader(2, "odd", 98304, 0, 0, 0, 0, 1, 0, 23, 0),
    ClassLeader(3, "even", 3108864, 0, 0, 0, 1, 0, 8, 0, 15),
    ClassLeader(3, "even", 5275648, 0, 0, 0, 0, 0, 12, 0, 12),
    ClassLeader(3, "odd", 98304, 0, 0, 1, 0, 0, 0, 23, 0),
    ClassLeader(3, "odd", 8290304, 0, 0, 0, 0, 3, 0, 21, 0),
    ClassLeader(4, "even", 170016, 0, 0, 0, 4, 0, 0, 0, 20),
    ClassLeader(4, "even", 48, 1, 0, 0, 0, 0, 0, 0, 23),
    ClassLeader(4, "even", 46632960, 0, 0, 0, 2, 0, 8, 0, 14),
    ClassLeader(4, "even", 777216, 0, 1, 0, 0, 0, 7, 0, 16),
    ClassLeader(4, "even", 126615552, 0, 0, 0, 1, 0, 12, 0, 11),
    ClassLeader(4, "even", 24870912, 0, 0, 0, 0, 0, 16, 0, 8),
    ClassLeader(4, "odd", 24870912, 0, 0, 1, 0, 2, 0, 21, 0),
    ClassLeader(4, "odd", 174096384, 0, 0, 0, 0, 5, 0, 19, 0),
)


class LeechLatticeVectorQuantizerGpu:
    """
    Implicit shell -> class -> local-symmetry database for Leech codewords.

    The database stores shell offsets, class offsets, class leaders, compatible
    Golay words, sign counts, and permutation-coset counts. It does not expand
    all codewords into memory.
    """

    def __init__(
        self,
        max_shell: int = 4,
        leaders: Sequence[ClassLeader] | None = None,
        cache_dir: str | Path | None = ".llvq_cache",
        use_cache: bool = True,
        verbose: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if max_shell < 2:
            raise ValueError("max_shell must be at least 2")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self._binom = torch.tensor(
            [[comb(n, k) if k <= n else 0 for k in range(DIM + 1)] for n in range(DIM + 1)],
            dtype=torch.long,
            device=self.device,
        )
        if leaders is None:
            leaders = generate_class_leaders(
                max_shell,
                cache_dir=cache_dir,
                use_cache=use_cache,
                verbose=verbose,
            )
        available_shells = {row.shell for row in leaders}
        missing = [m for m in range(2, max_shell + 1) if m not in available_shells]
        if missing:
            raise ValueError(f"no class leaders are available for shells: {missing}")

        self.max_shell = max_shell
        self.golay_codewords = extended_golay_codewords()
        self.golay_index_by_codeword = {
            codeword: index for index, codeword in enumerate(self.golay_codewords)
        }
        self.classes_by_shell = {
            shell: tuple(
                sorted(
                    (row for row in leaders if row.shell == shell),
                    key=lambda row: (row.leader, 0 if row.parity == "even" else 1, row.count),
                )
            )
            for shell in range(2, max_shell + 1)
        }
        self.shell_offsets = self._build_shell_offsets()
        self.class_offsets = self._build_class_offsets()
        self.local_structures = self._build_local_structures()
        self.class_index_by_shell_and_key = self._build_class_index()
        self.shell_counts = {
            shell: sum(row.count for row in rows) for shell, rows in self.classes_by_shell.items()
        }
        self.cumulative_shell_counts = {
            shell: self.shell_offsets[shell] + self.shell_counts[shell]
            for shell in range(2, max_shell + 1)
        }
        self.total_count = self.cumulative_shell_counts[max_shell]
        self.shape_bits = ceil(log2(self.total_count)) if self.total_count > 1 else 1
        self._validate()
        self._subclasses: tuple[dict, ...] | None = None

    def rank_class(self, shell: int, class_index: int, class_local_index: int = 0) -> int:
        classes = self.classes_by_shell[shell]
        if class_index < 0 or class_index >= len(classes):
            raise IndexError(class_index)
        leader = classes[class_index]
        if class_local_index < 0 or class_local_index >= leader.count:
            raise IndexError(class_local_index)
        return self.shell_offsets[shell] + self.class_offsets[shell][class_index] + class_local_index

    def unrank(self, global_index: int) -> RankedCodeword:
        if global_index < 0 or global_index >= self.total_count:
            raise IndexError(global_index)

        shell = self.shell_for_index(global_index)
        shell_local = global_index - self.shell_offsets[shell]
        offsets = self.class_offsets[shell]
        class_index = bisect_right(offsets, shell_local) - 1
        class_offset = offsets[class_index]
        return RankedCodeword(
            global_index=global_index,
            shell=shell,
            shell_local_index=shell_local,
            class_index=class_index,
            class_local_index=shell_local - class_offset,
            class_offset=class_offset,
            leader=self.classes_by_shell[shell][class_index],
        )

    def shell_for_index(self, global_index: int) -> int:
        ends = [self.cumulative_shell_counts[shell] for shell in range(2, self.max_shell + 1)]
        return 2 + bisect_right(ends, global_index)

    def decompose_local_index(self, ranked: RankedCodeword) -> LocalDecomposition:
        structure = self.local_structures[ranked.shell][ranked.class_index]
        cursor = ranked.class_local_index
        permutation_choice = cursor % structure.permutation_count
        cursor //= structure.permutation_count
        sign_choice = cursor % structure.sign_count
        cursor //= structure.sign_count
        golay_choice = cursor
        golay_index = structure.golay_codeword_indices[golay_choice]
        return LocalDecomposition(
            golay_choice=golay_choice,
            golay_codeword=self.golay_codewords[golay_index],
            sign_choice=sign_choice,
            permutation_choice=permutation_choice,
        )

    def codeword(self, global_index: int) -> tuple[int, ...]:
        ranked = self.unrank(global_index)
        local = self.decompose_local_index(ranked)
        return unrank_class_codeword(ranked.leader, local)

    def dequantize_lattice(self, global_index: int | Sequence[int] | torch.Tensor) -> tuple[int, ...] | torch.Tensor:
        """Return integer Leech representative(s) before the 1/sqrt(8) scale."""
        if not torch.is_tensor(global_index) and isinstance(global_index, int):
            return self.codeword(global_index)

        indices = torch.as_tensor(global_index, dtype=torch.long, device=self.device)
        output_shape = indices.shape
        codewords = self._dequantize_lattice_gpu(indices.reshape(-1))
        return codewords.reshape(*output_shape, DIM)

    def dequantize_lattice_original(
        self,
        global_index: int | Sequence[int] | torch.Tensor,
    ) -> tuple[int, ...] | torch.Tensor:
        """Reference Torch/Python batched dequantizer."""
        if not torch.is_tensor(global_index) and isinstance(global_index, int):
            return self.codeword(global_index)

        indices = torch.as_tensor(global_index, dtype=torch.long, device=self.device)
        output_shape = indices.shape
        codewords = self._dequantize_lattice_torch(indices.reshape(-1))
        return codewords.reshape(*output_shape, DIM)

    def dequantize_lattice_gpu(
        self,
        global_index: int | Sequence[int] | torch.Tensor,
    ) -> tuple[int, ...] | torch.Tensor:
        """Optimized GPU batched dequantizer used by `dequantize_lattice`."""
        if not torch.is_tensor(global_index) and isinstance(global_index, int):
            return self.codeword(global_index)

        indices = torch.as_tensor(global_index, dtype=torch.long, device=self.device)
        output_shape = indices.shape
        codewords = self._dequantize_lattice_gpu(indices.reshape(-1))
        return codewords.reshape(*output_shape, DIM)

    def dequantize_lattice_cuda(
        self,
        global_index: int | Sequence[int] | torch.Tensor,
    ) -> tuple[int, ...] | torch.Tensor:
        """NVRTC CUDA-kernel dequantizer; falls back only through `dequantize_lattice_gpu`."""
        return self.dequantize_lattice_gpu(global_index)

    def dequantize_lattice_triton(
        self,
        global_index: int | Sequence[int] | torch.Tensor,
    ) -> tuple[int, ...] | torch.Tensor:
        """Compatibility alias for the optimized GPU dequantizer."""
        return self.dequantize_lattice_gpu(global_index)

    def dequantize(self, global_index: int | Sequence[int] | torch.Tensor) -> tuple[float, ...] | torch.Tensor:
        """Return scaled Leech lattice vector(s) in Lambda_24."""
        if not torch.is_tensor(global_index) and isinstance(global_index, int):
            return tuple(LATTICE_SCALE * value for value in self.dequantize_lattice(global_index))

        codewords = self.dequantize_lattice(global_index)
        if not torch.is_tensor(codewords):
            return tuple(LATTICE_SCALE * value for value in codewords)
        return LATTICE_SCALE * codewords.to(self.dtype)

    def encode_codeword(self, vector: Sequence[int]) -> int:
        """
        Return the exact database index for a Leech codeword.

        This is the inverse of `dequantize` for vectors generated by this
        database. It validates shell, class leader, Golay compatibility,
        signs, and permutation rank. It does not perform nearest-neighbor
        search for arbitrary real-valued inputs.
        """
        values = tuple(int(value) for value in vector)
        if len(values) != DIM:
            raise ValueError(f"expected {DIM} coordinates, got {len(values)}")

        square_sum = sum(value * value for value in values)
        if square_sum % 16 != 0:
            raise ValueError(f"vector squared norm {square_sum} is not a Leech shell norm")
        shell = square_sum // 16
        if shell < 2 or shell > self.max_shell:
            raise ValueError(f"shell {shell} is outside database range 2..{self.max_shell}")

        parity = "even" if all(value % 2 == 0 for value in values) else "odd"
        if parity == "odd" and not all(value % 2 != 0 for value in values):
            raise ValueError("coordinates must be all even or all odd")

        leader_key = tuple(sorted((abs(value) for value in values), reverse=True))
        class_key = (leader_key, parity)
        try:
            class_index = self.class_index_by_shell_and_key[shell][class_key]
        except KeyError as exc:
            raise ValueError(f"no generated class matches shell={shell}, parity={parity}, leader={leader_key}") from exc

        structure = self.local_structures[shell][class_index]
        local = rank_class_codeword(values, structure, self.golay_index_by_codeword)
        return self.rank_class(shell, class_index, local)

    def quantize(self, vector: Sequence[float] | torch.Tensor) -> int | torch.Tensor:
        """
        Quantize real-valued 24-D vector(s) by brute-force Adoul-Barth subclass search.

        The search maximizes the projection x^T y over all generated shells,
        classes, and Golay subclasses. Each subclass is solved as a permutation
        code with the even/odd sign rules from the Leech construction.

        Tensor inputs may have shape `(..., 24)` and are searched/ranked on
        `self.device`. Sequence inputs preserve the CPU API and return `int`.
        """
        return self.quantize_cuda(vector)

    def quantize_cpu(self, vector: Sequence[float] | torch.Tensor) -> int | torch.Tensor:
        """
        Quantize real-valued 24-D vector(s) by Adoul-Barth subclass search.

        The search maximizes the projection x^T y over all generated shells,
        classes, and Golay subclasses. Each subclass is solved as a permutation
        code with the even/odd sign rules from the Leech construction.

        Tensor inputs may have shape `(..., 24)` and are searched/ranked on
        `self.device`. Sequence inputs preserve the CPU API and return `int`.
        """
        tensor_input = torch.is_tensor(vector)
        x = torch.as_tensor(vector, dtype=self.dtype, device=self.device)
        if x.shape[-1] != DIM:
            raise ValueError(f"expected last dimension {DIM}, got {x.shape[-1]}")

        single = x.ndim == 1
        output_shape = x.shape[:-1]
        x = x.reshape(-1, DIM).contiguous()
        rows = torch.arange(x.shape[0], device=self.device)

        best_score = torch.full((x.shape[0],), float("-inf"), dtype=self.dtype, device=self.device)
        best_codeword = torch.zeros((x.shape[0], DIM), dtype=torch.long, device=self.device)
        best_shell = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        best_class_index = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        best_golay_choice = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)

        for golay_index, codeword in enumerate(self.golay_codewords):
            for shell in range(2, self.max_shell + 1):
                for structure in self.local_structures[shell]:
                    if structure.leader.parity == "even":
                        if sum(codeword) != structure.leader.even_weight:
                            continue
                        candidate, projection = self._solve_even_subclass_gpu(x, structure.leader, codeword)
                    else:
                        candidate, projection = self._solve_odd_subclass_gpu(x, structure.leader, codeword)

                    try:
                        golay_choice = structure.golay_codeword_indices.index(golay_index)
                    except ValueError:
                        continue

                    score = LATTICE_SCALE * projection - shell
                    improved = score > best_score
                    if improved.any():
                        best_score[improved] = score[improved]
                        best_codeword[improved] = candidate[improved]
                        best_shell[improved] = shell
                        best_class_index[improved] = structure.class_index
                        best_golay_choice[improved] = golay_choice

        if (best_score == float("-inf")).any():
            raise RuntimeError("no Leech candidate found")

        indices = self._rank_codewords_gpu(
            best_codeword,
            best_shell,
            best_class_index,
            best_golay_choice,
        ).reshape(output_shape)
        if single:
            indices = indices.reshape(())
        if not tensor_input:
            return int(indices.item())
        return indices

    def quantize_optimized(
        self,
        vector: Sequence[float] | torch.Tensor,
        subclass_chunk_size: int = 512,
    ) -> int | torch.Tensor:
        """
        Quantize by evaluating many Golay/shell/class subclasses in parallel.

        This keeps the exact same candidate construction and ranking as
        `quantize`, but flattens valid `(golay, shell, class)` combinations into
        metadata chunks and scores a whole `[batch, subclass_chunk]` tile on GPU.
        """
        tensor_input = torch.is_tensor(vector)
        x = torch.as_tensor(vector, dtype=self.dtype, device=self.device)
        if x.shape[-1] != DIM:
            raise ValueError(f"expected last dimension {DIM}, got {x.shape[-1]}")

        single = x.ndim == 1
        output_shape = x.shape[:-1]
        x = x.reshape(-1, DIM).contiguous()

        best_score = torch.full((x.shape[0],), float("-inf"), dtype=self.dtype, device=self.device)
        best_codeword = torch.zeros((x.shape[0], DIM), dtype=torch.long, device=self.device)
        best_shell = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        best_class_index = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        best_golay_choice = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        subclasses = self._build_subclasses()
        best_order = torch.full((x.shape[0],), len(subclasses) + 1, dtype=torch.long, device=self.device)

        for start in range(0, len(subclasses), subclass_chunk_size):
            chunk = subclasses[start : start + subclass_chunk_size]
            for parity in ("even", "odd"):
                parity_chunk = [item for item in chunk if item["parity"] == parity]
                if not parity_chunk:
                    continue
                if parity == "even":
                    candidates, scores = self._solve_even_subclass_chunk_gpu(x, parity_chunk)
                else:
                    candidates, scores = self._solve_odd_subclass_chunk_gpu(x, parity_chunk)

                order = torch.tensor([item["order"] for item in parity_chunk], dtype=torch.long, device=self.device)
                _, chunk_pos = scores.max(dim=1)
                rows = torch.arange(x.shape[0], device=self.device)
                chunk_scores = scores[rows, chunk_pos]
                chunk_orders = order[chunk_pos]
                improved = (chunk_scores > best_score) | (
                    (chunk_scores == best_score) & (chunk_orders < best_order)
                )
                if not improved.any():
                    continue

                best_score[improved] = chunk_scores[improved]
                best_order[improved] = chunk_orders[improved]
                best_codeword[improved] = candidates[rows[improved], chunk_pos[improved]]

                shells = torch.tensor([item["shell"] for item in parity_chunk], dtype=torch.long, device=self.device)
                class_indices = torch.tensor(
                    [item["class_index"] for item in parity_chunk],
                    dtype=torch.long,
                    device=self.device,
                )
                golay_choices = torch.tensor(
                    [item["golay_choice"] for item in parity_chunk],
                    dtype=torch.long,
                    device=self.device,
                )
                best_shell[improved] = shells[chunk_pos[improved]]
                best_class_index[improved] = class_indices[chunk_pos[improved]]
                best_golay_choice[improved] = golay_choices[chunk_pos[improved]]

        if (best_score == float("-inf")).any():
            raise RuntimeError("no Leech candidate found")

        indices = self._rank_codewords_gpu(
            best_codeword,
            best_shell,
            best_class_index,
            best_golay_choice,
        ).reshape(output_shape)

        if single:
            indices = indices.reshape(())
        if not tensor_input:
            return int(indices.item())
        return indices

    def quantize_triton(
        self,
        vector: Sequence[float] | torch.Tensor,
        subclass_chunk_size: int = 32768,
    ) -> int | torch.Tensor:
        """
        Quantize with Triton score-only kernels.

        Triton computes only best subclass scores/metadata. Candidate codewords
        and exact hierarchical indices are reconstructed only for the winning
        subclass of each input vector.
        """
        if self.device.type != "cuda":
            raise ValueError("quantize_triton requires a CUDA device")

        tensor_input = torch.is_tensor(vector)
        x = torch.as_tensor(vector, dtype=self.dtype, device=self.device)
        if x.shape[-1] != DIM:
            raise ValueError(f"expected last dimension {DIM}, got {x.shape[-1]}")

        single = x.ndim == 1
        output_shape = x.shape[:-1]
        x = x.reshape(-1, DIM).contiguous()

        meta = self._compact_triton_metadata()
        best_score = torch.full((x.shape[0],), float("-inf"), dtype=torch.float32, device=self.device)
        best_order = torch.full((x.shape[0],), meta["total_pairs"] + 1, dtype=torch.long, device=self.device)
        best_pair = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)

        total_pairs = int(meta["total_pairs"])
        for start in range(0, total_pairs, subclass_chunk_size):
            chunk_n = min(subclass_chunk_size, total_pairs - start)
            scores = torch.empty((x.shape[0], chunk_n), dtype=torch.float32, device=self.device)
            _score_compact_pairs_kernel[(x.shape[0], chunk_n)](
                x,
                meta["golay_bits"],
                meta["golay_weight"],
                meta["class_parity"],
                meta["class_shell"],
                meta["class_even_weight"],
                meta["class_f0_mags"],
                meta["class_f1_mags"],
                meta["class_f0_len"],
                meta["class_f1_len"],
                meta["class_required"],
                meta["class_odd_leaders"],
                scores,
                start,
                chunk_n,
                meta["n_classes"],
                meta["n_golay"],
                DIM,
            )

            chunk_scores, chunk_pos = scores.max(dim=1)
            pair = chunk_pos + start
            order = (pair % meta["n_golay"]) * meta["n_classes"] + torch.div(
                pair,
                meta["n_golay"],
                rounding_mode="floor",
            )
            improved = (chunk_scores > best_score) | (
                (chunk_scores == best_score) & (order < best_order)
            )
            best_score[improved] = chunk_scores[improved]
            best_order[improved] = order[improved]
            best_pair[improved] = pair[improved]

        codewords, shells, class_indices, golay_choices = self._codewords_for_compact_pairs(
            x,
            best_pair,
        )
        indices = self._rank_codewords_gpu(codewords, shells, class_indices, golay_choices).reshape(output_shape)
        if single:
            indices = indices.reshape(())
        if not tensor_input:
            return int(indices.item())
        return indices

    def quantize_cuda(self, vector: Sequence[float] | torch.Tensor) -> int | torch.Tensor:
        """
        Quantize with the fused NVRTC CUDA kernel.

        CUDA computes the winning compact `(class, Golay)` pair, reconstructs
        the winning codeword, ranks it, and returns final database indices.
        """
        if self.device.type != "cuda":
            raise ValueError("quantize_cuda requires a CUDA device")
        if quantize_lattice_cuda is None:
            return self.quantize_triton(vector)

        tensor_input = torch.is_tensor(vector)
        x = torch.as_tensor(vector, dtype=self.dtype, device=self.device)
        if x.shape[-1] != DIM:
            raise ValueError(f"expected last dimension {DIM}, got {x.shape[-1]}")

        single = x.ndim == 1
        output_shape = x.shape[:-1]
        x = x.reshape(-1, DIM).contiguous()
        x_scores = x.to(torch.float32).contiguous()

        meta = self._compact_triton_metadata()
        rank_meta = self._dequantize_triton_metadata()
        indices = torch.empty((x.shape[0],), dtype=torch.long, device=self.device)
        best_pair = torch.empty((x.shape[0],), dtype=torch.long, device=self.device)
        best_score = torch.empty((x.shape[0],), dtype=torch.float32, device=self.device)
        quantize_lattice_cuda(x_scores, indices, best_pair, best_score, meta, rank_meta, self._binom)

        indices = indices.reshape(output_shape)
        if single:
            indices = indices.reshape(())
        if not tensor_input:
            return int(indices.item())
        return indices

    def _triton_metadata(self) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        cached = getattr(self, "_triton_meta", None)
        if cached is not None:
            return cached

        subclasses = self._build_subclasses()
        order_by_global_pos = torch.tensor(
            [item["order"] for item in subclasses],
            dtype=torch.long,
            device=self.device,
        )
        meta: dict[str, dict[str, torch.Tensor] | torch.Tensor] = {
            "order_by_global_pos": order_by_global_pos,
        }

        for parity in ("even", "odd"):
            items = [(pos, item) for pos, item in enumerate(subclasses) if item["parity"] == parity]
            if not items:
                meta[parity] = {
                    "order": torch.empty(0, dtype=torch.long, device=self.device),
                    "global_pos": torch.empty(0, dtype=torch.long, device=self.device),
                    "shell": torch.empty(0, dtype=torch.float32, device=self.device),
                }
                continue

            global_pos = torch.tensor([pos for pos, _ in items], dtype=torch.long, device=self.device)
            order = torch.tensor([item["order"] for _, item in items], dtype=torch.long, device=self.device)
            shell = torch.tensor([item["shell"] for _, item in items], dtype=torch.float32, device=self.device)
            if parity == "odd":
                bits = torch.tensor(
                    [
                        sum(1 << pos for pos, bit in enumerate(item["golay_codeword"]) if bit)
                        for _, item in items
                    ],
                    dtype=torch.int32,
                    device=self.device,
                )
                signed_leaders = torch.tensor(
                    [sorted(_odd_signed_leader_values(item["leader"])) for _, item in items],
                    dtype=torch.int32,
                    device=self.device,
                )
                meta[parity] = {
                    "global_pos": global_pos,
                    "order": order,
                    "shell": shell,
                    "bits": bits,
                    "signed_leaders": signed_leaders,
                }
            else:
                f0_mask_values: list[int] = []
                f1_mask_values: list[int] = []
                f0_mags_values: list[tuple[int, ...]] = []
                f1_mags_values: list[tuple[int, ...]] = []
                f0_len_values: list[int] = []
                f1_len_values: list[int] = []
                required_values: list[int] = []
                for _, item in items:
                    codeword = item["golay_codeword"]
                    leader = item["leader"]
                    f0_values = _ascending_multiset(_even_f0_multiplicities(leader))
                    f1_values = _ascending_multiset(_even_f1_multiplicities(leader))
                    f0_len_values.append(len(f0_values))
                    f1_len_values.append(len(f1_values))
                    required_values.append((sum(leader.leader) % 8) // 4)
                    f0_mask = 0
                    f1_mask = 0
                    for pos, bit in enumerate(codeword):
                        if bit == 0:
                            f0_mask |= 1 << pos
                        else:
                            f1_mask |= 1 << pos
                    f0_mask_values.append(f0_mask)
                    f1_mask_values.append(f1_mask)
                    f0_mags_values.append(f0_values + (0,) * (DIM - len(f0_values)))
                    f1_mags_values.append(f1_values + (0,) * (DIM - len(f1_values)))
                meta[parity] = {
                    "global_pos": global_pos,
                    "order": order,
                    "shell": shell,
                    "f0_mask": torch.tensor(f0_mask_values, dtype=torch.int32, device=self.device),
                    "f1_mask": torch.tensor(f1_mask_values, dtype=torch.int32, device=self.device),
                    "f0_mags": torch.tensor(f0_mags_values, dtype=torch.int32, device=self.device),
                    "f1_mags": torch.tensor(f1_mags_values, dtype=torch.int32, device=self.device),
                    "f0_len": torch.tensor(f0_len_values, dtype=torch.int32, device=self.device),
                    "f1_len": torch.tensor(f1_len_values, dtype=torch.int32, device=self.device),
                    "required": torch.tensor(required_values, dtype=torch.int32, device=self.device),
                }

        self._triton_meta = meta
        return meta

    def _compact_triton_metadata(self) -> dict[str, torch.Tensor | int | tuple[ClassLocalStructure, ...]]:
        cached = getattr(self, "_compact_triton_meta", None)
        if cached is not None:
            return cached

        structures: list[ClassLocalStructure] = []
        parity: list[int] = []
        shell: list[float] = []
        even_weight: list[int] = []
        f0_mags: list[tuple[int, ...]] = []
        f1_mags: list[tuple[int, ...]] = []
        f0_len: list[int] = []
        f1_len: list[int] = []
        required: list[int] = []
        odd_leaders: list[tuple[int, ...]] = []

        for shell_id in range(2, self.max_shell + 1):
            for structure in self.local_structures[shell_id]:
                leader = structure.leader
                structures.append(structure)
                parity.append(0 if leader.parity == "even" else 1)
                shell.append(float(shell_id))
                even_weight.append(leader.even_weight if leader.parity == "even" else -1)
                if leader.parity == "even":
                    f0_values = _ascending_multiset(_even_f0_multiplicities(leader))
                    f1_values = _ascending_multiset(_even_f1_multiplicities(leader))
                    odd_values: tuple[int, ...] = (0,) * DIM
                else:
                    f0_values = ()
                    f1_values = ()
                    odd_values = tuple(sorted(_odd_signed_leader_values(leader)))
                f0_len.append(len(f0_values))
                f1_len.append(len(f1_values))
                required.append((sum(leader.leader) % 8) // 4 if leader.parity == "even" else 0)
                f0_mags.append(f0_values + (0,) * (DIM - len(f0_values)))
                f1_mags.append(f1_values + (0,) * (DIM - len(f1_values)))
                odd_leaders.append(odd_values)

        golay_bits: list[int] = []
        golay_weight: list[int] = []
        for codeword in self.golay_codewords:
            mask = 0
            weight = 0
            for bit, value in enumerate(codeword):
                if value:
                    mask |= 1 << bit
                    weight += 1
            golay_bits.append(mask)
            golay_weight.append(weight)

        n_classes = len(structures)
        n_golay = len(self.golay_codewords)
        self._compact_triton_meta = {
            "structures": tuple(structures),
            "n_classes": n_classes,
            "n_golay": n_golay,
            "total_pairs": n_classes * n_golay,
            "golay_bits": torch.tensor(golay_bits, dtype=torch.int32, device=self.device),
            "golay_weight": torch.tensor(golay_weight, dtype=torch.int32, device=self.device),
            "class_parity": torch.tensor(parity, dtype=torch.int32, device=self.device),
            "class_shell": torch.tensor(shell, dtype=torch.float32, device=self.device),
            "class_even_weight": torch.tensor(even_weight, dtype=torch.int32, device=self.device),
            "class_f0_mags": torch.tensor(f0_mags, dtype=torch.int32, device=self.device),
            "class_f1_mags": torch.tensor(f1_mags, dtype=torch.int32, device=self.device),
            "class_f0_len": torch.tensor(f0_len, dtype=torch.int32, device=self.device),
            "class_f1_len": torch.tensor(f1_len, dtype=torch.int32, device=self.device),
            "class_required": torch.tensor(required, dtype=torch.int32, device=self.device),
            "class_odd_leaders": torch.tensor(odd_leaders, dtype=torch.int32, device=self.device),
        }
        return self._compact_triton_meta

    def _codewords_for_compact_pairs(
        self,
        x: torch.Tensor,
        pairs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        meta = self._compact_triton_metadata()
        structures = meta["structures"]
        n_golay = int(meta["n_golay"])
        out = torch.zeros((x.shape[0], DIM), dtype=torch.long, device=self.device)
        shells = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        class_indices = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        golay_choices = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)

        for pair in pairs.unique().detach().cpu().tolist():
            class_id = int(pair) // n_golay
            golay_index = int(pair) % n_golay
            structure = structures[class_id]
            try:
                golay_choice = structure.golay_codeword_indices.index(golay_index)
            except ValueError as exc:
                raise RuntimeError("winner selected an incompatible Golay/class pair") from exc
            mask = pairs == pair
            if structure.leader.parity == "even":
                cand, _ = self._solve_even_subclass_gpu(
                    x[mask],
                    structure.leader,
                    self.golay_codewords[golay_index],
                )
            else:
                cand, _ = self._solve_odd_subclass_gpu(
                    x[mask],
                    structure.leader,
                    self.golay_codewords[golay_index],
                )
            out[mask] = cand
            shells[mask] = structure.shell
            class_indices[mask] = structure.class_index
            golay_choices[mask] = golay_choice
        return out, shells, class_indices, golay_choices

    def _codewords_for_global_subclass_positions(
        self,
        x: torch.Tensor,
        global_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        subclasses = self._build_subclasses()
        out = torch.zeros((x.shape[0], DIM), dtype=torch.long, device=self.device)
        shells = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        class_indices = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)
        golay_choices = torch.zeros((x.shape[0],), dtype=torch.long, device=self.device)

        for pos in global_positions.unique().detach().cpu().tolist():
            mask = global_positions == pos
            item = subclasses[int(pos)]
            if item["parity"] == "even":
                cand, _ = self._solve_even_subclass_gpu(x[mask], item["leader"], item["golay_codeword"])
            else:
                cand, _ = self._solve_odd_subclass_gpu(x[mask], item["leader"], item["golay_codeword"])
            out[mask] = cand
            shells[mask] = item["shell"]
            class_indices[mask] = item["class_index"]
            golay_choices[mask] = item["golay_choice"]
        return out, shells, class_indices, golay_choices

    def _build_subclasses(self) -> tuple[dict, ...]:
        if self._subclasses is not None:
            return self._subclasses

        subclasses = []
        for shell in range(2, self.max_shell + 1):
            for structure in self.local_structures[shell]:
                for golay_choice, golay_index in enumerate(structure.golay_codeword_indices):
                    codeword = self.golay_codewords[golay_index]
                    subclasses.append(
                        {
                            "shell": shell,
                            "class_index": structure.class_index,
                            "golay_choice": golay_choice,
                            "golay_index": golay_index,
                            "golay_codeword": codeword,
                            "leader": structure.leader,
                            "parity": structure.leader.parity,
                        }
                    )
        subclasses.sort(key=lambda item: (item["golay_index"], item["shell"], item["class_index"]))
        for order, item in enumerate(subclasses):
            item["order"] = order
        self._subclasses = tuple(subclasses)
        return self._subclasses

    def _solve_odd_subclass_chunk_gpu(
        self,
        x: torch.Tensor,
        chunk: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codewords = torch.tensor(
            [item["golay_codeword"] for item in chunk],
            dtype=torch.long,
            device=self.device,
        )
        flips = torch.where(
            codewords == 1,
            -torch.ones((), dtype=self.dtype, device=self.device),
            torch.ones((), dtype=self.dtype, device=self.device),
        )
        x_prime = x[:, None, :] * flips[None, :, :]
        ordered_positions = torch.argsort(x_prime, dim=2, stable=True)
        signed_leaders = torch.tensor(
            [sorted(_odd_signed_leader_values(item["leader"])) for item in chunk],
            dtype=torch.long,
            device=self.device,
        )
        candidates_prime = torch.zeros((x.shape[0], len(chunk), DIM), dtype=torch.long, device=self.device)
        src = signed_leaders[None, :, :].expand(x.shape[0], -1, -1)
        candidates_prime.scatter_(2, ordered_positions, src)
        candidates = candidates_prime * flips.to(torch.long)[None, :, :]
        projection = (x[:, None, :] * candidates.to(self.dtype)).sum(dim=2)
        shells = torch.tensor([item["shell"] for item in chunk], dtype=self.dtype, device=self.device)
        scores = LATTICE_SCALE * projection - shells[None, :]
        return candidates, scores

    def _solve_even_subclass_chunk_gpu(
        self,
        x: torch.Tensor,
        chunk: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        n = len(chunk)
        candidates = torch.zeros((batch, n, DIM), dtype=torch.long, device=self.device)
        abs_x = x.abs()

        max_f0 = max(len([bit for bit in item["golay_codeword"] if bit == 0]) for item in chunk)
        max_f1 = max(len([bit for bit in item["golay_codeword"] if bit == 1]) for item in chunk)
        f0_masks = torch.zeros((n, DIM), dtype=torch.bool, device=self.device)
        f1_masks = torch.zeros((n, DIM), dtype=torch.bool, device=self.device)
        f0_mags = torch.zeros((n, max_f0), dtype=torch.long, device=self.device)
        f1_mags = torch.zeros((n, max_f1), dtype=torch.long, device=self.device)
        f0_len = torch.zeros((n,), dtype=torch.long, device=self.device)
        f1_len = torch.zeros((n,), dtype=torch.long, device=self.device)
        required_f1_negative_parity = torch.zeros((n,), dtype=torch.long, device=self.device)

        for i, item in enumerate(chunk):
            codeword = item["golay_codeword"]
            leader = item["leader"]
            f0_positions = [pos for pos, bit in enumerate(codeword) if bit == 0]
            f1_positions = [pos for pos, bit in enumerate(codeword) if bit == 1]
            f0_values = _ascending_multiset(_even_f0_multiplicities(leader))
            f1_values = _ascending_multiset(_even_f1_multiplicities(leader))
            f0_masks[i, f0_positions] = True
            f1_masks[i, f1_positions] = True
            f0_len[i] = len(f0_values)
            f1_len[i] = len(f1_values)
            if f0_values:
                f0_mags[i, : len(f0_values)] = torch.tensor(f0_values, dtype=torch.long, device=self.device)
            if f1_values:
                f1_mags[i, : len(f1_values)] = torch.tensor(f1_values, dtype=torch.long, device=self.device)
            required_f1_negative_parity[i] = (sum(leader.leader) % 8) // 4

        self._scatter_even_chunk_group(abs_x, x, candidates, f0_masks, f0_mags, f0_len)
        ordered_f1 = self._scatter_even_chunk_group(abs_x, x, candidates, f1_masks, f1_mags, f1_len)

        has_f1 = f1_len > 0
        if has_f1.any():
            actual_negative_parity = ((candidates < 0) & f1_masks[None, :, :]).sum(dim=2).remainder(2)
            needs_flip = has_f1[None, :] & (actual_negative_parity != required_f1_negative_parity[None, :])
            if needs_flip.any():
                rows, cols = needs_flip.nonzero(as_tuple=True)
                flip_positions = ordered_f1[rows, cols, 0]
                candidates[rows, cols, flip_positions] *= -1

        projection = (x[:, None, :] * candidates.to(self.dtype)).sum(dim=2)
        shells = torch.tensor([item["shell"] for item in chunk], dtype=self.dtype, device=self.device)
        scores = LATTICE_SCALE * projection - shells[None, :]
        return candidates, scores

    def _scatter_even_chunk_group(
        self,
        abs_x: torch.Tensor,
        x: torch.Tensor,
        candidates: torch.Tensor,
        masks: torch.Tensor,
        magnitudes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_len = magnitudes.shape[1]
        if max_len == 0:
            return torch.empty((*candidates.shape[:2], 0), dtype=torch.long, device=self.device)

        values = abs_x[:, None, :].expand(-1, masks.shape[0], -1).masked_fill(~masks[None, :, :], torch.inf)
        ordered_positions = torch.argsort(values, dim=2, stable=True)[:, :, :max_len]
        take = torch.arange(max_len, device=self.device)[None, None, :] < lengths[None, :, None]
        selected_x = x[:, None, :].expand(-1, masks.shape[0], -1).gather(2, ordered_positions)
        signs = torch.where(
            selected_x >= 0,
            torch.ones((), dtype=torch.long, device=self.device),
            -torch.ones((), dtype=torch.long, device=self.device),
        )
        src = signs * magnitudes[None, :, :]
        update = torch.zeros_like(candidates)
        update_mask = torch.zeros_like(candidates, dtype=torch.bool)
        update.scatter_(2, ordered_positions, src)
        update_mask.scatter_(2, ordered_positions, take.expand(candidates.shape[0], -1, -1))
        candidates.copy_(torch.where(update_mask, update, candidates))
        return ordered_positions

    def _solve_even_subclass_gpu(
        self,
        x: torch.Tensor,
        leader: ClassLeader,
        golay_codeword: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate = torch.zeros((x.shape[0], DIM), dtype=torch.long, device=self.device)
        f0_positions = tuple(i for i, bit in enumerate(golay_codeword) if bit == 0)
        f1_positions = tuple(i for i, bit in enumerate(golay_codeword) if bit == 1)
        f0_magnitudes = _ascending_multiset(_even_f0_multiplicities(leader))
        f1_magnitudes = _ascending_multiset(_even_f1_multiplicities(leader))

        self._scatter_even_group(x, candidate, f0_positions, f0_magnitudes)
        ordered_f1 = self._scatter_even_group(x, candidate, f1_positions, f1_magnitudes)

        if f1_magnitudes:
            required_negative_parity = (sum(leader.leader) % 8) // 4
            actual_negative_parity = (candidate[:, list(f1_positions)] < 0).sum(dim=1).remainder(2)
            needs_flip = actual_negative_parity != required_negative_parity
            if needs_flip.any():
                rows = torch.arange(x.shape[0], device=self.device)
                candidate[rows[needs_flip], ordered_f1[needs_flip, 0]] *= -1

        projection = (x * candidate.to(self.dtype)).sum(dim=1)
        return candidate, projection

    def _scatter_even_group(
        self,
        x: torch.Tensor,
        candidate: torch.Tensor,
        positions: tuple[int, ...],
        magnitudes: tuple[int, ...],
    ) -> torch.Tensor:
        if not positions:
            return torch.empty((x.shape[0], 0), dtype=torch.long, device=self.device)

        pos = torch.tensor(positions, dtype=torch.long, device=self.device)
        mags = torch.tensor(magnitudes, dtype=torch.long, device=self.device)
        order = torch.argsort(x[:, pos].abs(), dim=1, stable=True)
        ordered_positions = pos[order]
        selected_x = x.gather(1, ordered_positions)
        signs = torch.where(
            selected_x >= 0,
            torch.ones((), dtype=torch.long, device=self.device),
            -torch.ones((), dtype=torch.long, device=self.device),
        )
        candidate.scatter_(1, ordered_positions, signs * mags.unsqueeze(0))
        return ordered_positions

    def _solve_odd_subclass_gpu(
        self,
        x: torch.Tensor,
        leader: ClassLeader,
        golay_codeword: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codeword = torch.tensor(golay_codeword, dtype=torch.long, device=self.device)
        flip = torch.where(
            codeword == 1,
            -torch.ones((), dtype=self.dtype, device=self.device),
            torch.ones((), dtype=self.dtype, device=self.device),
        )
        x_prime = x * flip
        ordered_positions = torch.argsort(x_prime, dim=1, stable=True)
        signed_leader = torch.tensor(sorted(_odd_signed_leader_values(leader)), dtype=torch.long, device=self.device)
        candidate_prime = torch.zeros((x.shape[0], DIM), dtype=torch.long, device=self.device)
        candidate_prime.scatter_(1, ordered_positions, signed_leader.unsqueeze(0).expand_as(candidate_prime))
        candidate = candidate_prime * flip.to(torch.long).unsqueeze(0)
        projection = (x * candidate.to(self.dtype)).sum(dim=1)
        return candidate, projection

    def _rank_codewords_gpu(
        self,
        codewords: torch.Tensor,
        shells: torch.Tensor,
        class_indices: torch.Tensor,
        golay_choices: torch.Tensor,
    ) -> torch.Tensor:
        indices = torch.zeros((codewords.shape[0],), dtype=torch.long, device=self.device)
        for shell in range(2, self.max_shell + 1):
            for structure in self.local_structures[shell]:
                class_mask = (shells == shell) & (class_indices == structure.class_index)
                if not class_mask.any():
                    continue
                for golay_choice in range(len(structure.golay_codeword_indices)):
                    mask = class_mask & (golay_choices == golay_choice)
                    if not mask.any():
                        continue

                    vectors = codewords[mask]
                    if structure.leader.parity == "even":
                        local = self._rank_even_codewords_gpu(vectors, structure, golay_choice)
                    else:
                        local = self._rank_odd_codewords_gpu(vectors, structure, golay_choice)

                    indices[mask] = (
                        self.shell_offsets[shell]
                        + self.class_offsets[shell][structure.class_index]
                        + local
                    )
        return indices

    def _dequantize_lattice_gpu(self, indices: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cuda" and dequantize_lattice_cuda is not None:
            return self._dequantize_lattice_cuda(indices)
        return self._dequantize_lattice_torch_fast(indices)

    def _dequantize_lattice_cuda(self, indices: torch.Tensor) -> torch.Tensor:
        if ((indices < 0) | (indices >= self.total_count)).any():
            raise IndexError("global index outside codebook range")
        meta = self._dequantize_triton_metadata()
        out = torch.empty((indices.shape[0], DIM), dtype=torch.long, device=self.device)
        try:
            return dequantize_lattice_cuda(indices.contiguous(), out, meta, self._binom)
        except Exception:
            return self._dequantize_lattice_torch_fast(indices)

    def _dequantize_lattice_torch_fast(self, indices: torch.Tensor) -> torch.Tensor:
        if ((indices < 0) | (indices >= self.total_count)).any():
            raise IndexError("global index outside codebook range")

        meta = self._decode_metadata()
        class_id = torch.bucketize(indices, meta["class_starts"], right=True) - 1
        local = indices - meta["class_starts"].gather(0, class_id)
        out = torch.empty((indices.shape[0], DIM), dtype=torch.long, device=self.device)

        for flat_class_id in torch.unique(class_id).detach().cpu().tolist():
            class_mask = class_id == flat_class_id
            structure = meta["flat_structures"][flat_class_id]
            class_local = local[class_mask]

            permutation_choice = class_local.remainder(structure.permutation_count)
            cursor = torch.div(class_local, structure.permutation_count, rounding_mode="floor")
            sign_choice = cursor.remainder(structure.sign_count)
            golay_choice = torch.div(cursor, structure.sign_count, rounding_mode="floor")

            class_out = torch.empty((class_local.shape[0], DIM), dtype=torch.long, device=self.device)
            for choice in torch.unique(golay_choice).detach().cpu().tolist():
                choice_mask = golay_choice == choice
                if structure.leader.parity == "even":
                    class_out[choice_mask] = self._unrank_even_codewords_gpu(
                        structure,
                        choice,
                        sign_choice[choice_mask],
                        permutation_choice[choice_mask],
                    )
                else:
                    class_out[choice_mask] = self._unrank_odd_codewords_gpu(
                        structure,
                        choice,
                        permutation_choice[choice_mask],
                    )

            out[class_mask] = class_out
        return out

    def _dequantize_lattice_torch(self, indices: torch.Tensor) -> torch.Tensor:
        if ((indices < 0) | (indices >= self.total_count)).any():
            raise IndexError("global index outside codebook range")

        meta = self._decode_metadata()
        out = torch.zeros((indices.shape[0], DIM), dtype=torch.long, device=self.device)
        shells = 2 + torch.bucketize(indices, meta["shell_ends"], right=True)

        for shell in range(2, self.max_shell + 1):
            shell_mask = shells == shell
            if not shell_mask.any():
                continue
            shell_indices = indices[shell_mask]
            shell_local = shell_indices - self.shell_offsets[shell]
            class_offsets = meta["class_offsets"][shell]
            class_indices = torch.bucketize(shell_local, class_offsets, right=True) - 1

            for structure in self.local_structures[shell]:
                class_mask = class_indices == structure.class_index
                if not class_mask.any():
                    continue

                local = shell_local[class_mask] - self.class_offsets[shell][structure.class_index]
                permutation_choice = local.remainder(structure.permutation_count)
                cursor = torch.div(local, structure.permutation_count, rounding_mode="floor")
                sign_choice = cursor.remainder(structure.sign_count)
                golay_choice = torch.div(cursor, structure.sign_count, rounding_mode="floor")

                class_out = torch.zeros((local.shape[0], DIM), dtype=torch.long, device=self.device)
                for choice in range(len(structure.golay_codeword_indices)):
                    choice_mask = golay_choice == choice
                    if not choice_mask.any():
                        continue
                    if structure.leader.parity == "even":
                        class_out[choice_mask] = self._unrank_even_codewords_gpu(
                            structure,
                            choice,
                            sign_choice[choice_mask],
                            permutation_choice[choice_mask],
                        )
                    else:
                        class_out[choice_mask] = self._unrank_odd_codewords_gpu(
                            structure,
                            choice,
                            permutation_choice[choice_mask],
                        )

                shell_rows = shell_mask.nonzero(as_tuple=False).flatten()
                out[shell_rows[class_mask]] = class_out
        return out

    def _dequantize_lattice_triton(self, indices: torch.Tensor) -> torch.Tensor:
        if ((indices < 0) | (indices >= self.total_count)).any():
            raise IndexError("global index outside codebook range")
        meta = self._dequantize_triton_metadata()
        class_id = torch.bucketize(indices, meta["class_starts"], right=True) - 1
        local = indices - meta["class_starts"].gather(0, class_id)
        out = torch.empty((indices.shape[0], DIM), dtype=torch.long, device=self.device)
        grid = (indices.shape[0],)
        _dequantize_lattice_even_kernel[grid](
            local.contiguous(),
            class_id.contiguous(),
            out,
            meta["parity"],
            meta["perm_count"],
            meta["sign_count"],
            meta["f1_perm"],
            meta["golay_start"],
            meta["golay_indices"],
            meta["golay_bits"],
            meta["f0_counts"],
            meta["f1_counts"],
            meta["f0_values"],
            meta["f1_values"],
            self._binom,
            indices.shape[0],
            DIM,
            32,
            DIM + 1,
            meta["max_vals"],
        )
        _dequantize_lattice_odd_kernel[grid](
            local.contiguous(),
            class_id.contiguous(),
            out,
            meta["parity"],
            meta["perm_count"],
            meta["golay_start"],
            meta["golay_indices"],
            meta["golay_bits"],
            meta["odd_counts"],
            meta["odd_values"],
            self._binom,
            indices.shape[0],
            DIM,
            32,
            DIM + 1,
            meta["max_vals"],
        )
        return out

    def _dequantize_triton_metadata(self) -> dict[str, torch.Tensor | int]:
        cached = getattr(self, "_dequantize_triton_meta", None)
        if cached is not None:
            return cached

        max_distinct = 1
        for shell in range(2, self.max_shell + 1):
            for structure in self.local_structures[shell]:
                leader = structure.leader
                if leader.parity == "even":
                    max_distinct = max(
                        max_distinct,
                        len(_even_f0_multiplicities(leader)),
                        len(_even_f1_multiplicities(leader)),
                    )
                else:
                    max_distinct = max(max_distinct, len(_odd_multiplicities(leader)))
        max_vals = 1 << (max_distinct - 1).bit_length()
        class_starts = []
        parity = []
        perm_count = []
        sign_count = []
        f1_perm = []
        golay_start = []
        golay_count = []
        golay_indices = []
        f0_counts = []
        f1_counts = []
        odd_counts = []
        f0_values = []
        f1_values = []
        odd_values = []

        def counts_and_values(multiplicities: dict[int, int]) -> tuple[list[int], list[int]]:
            values = [value for value, count in sorted(multiplicities.items(), reverse=True) if count]
            counts = [multiplicities[value] for value in values]
            values = values + [0] * (max_vals - len(values))
            counts = counts + [0] * (max_vals - len(counts))
            return counts[:max_vals], values[:max_vals]

        for shell in range(2, self.max_shell + 1):
            for structure in self.local_structures[shell]:
                leader = structure.leader
                class_starts.append(self.shell_offsets[shell] + self.class_offsets[shell][structure.class_index])
                parity.append(0 if leader.parity == "even" else 1)
                perm_count.append(structure.permutation_count)
                sign_count.append(structure.sign_count)
                golay_start.append(len(golay_indices))
                golay_count.append(len(structure.golay_codeword_indices))
                golay_indices.extend(structure.golay_codeword_indices)
                if leader.parity == "even":
                    f0c, f0v = counts_and_values(_even_f0_multiplicities(leader))
                    f1c, f1v = counts_and_values(_even_f1_multiplicities(leader))
                    oddc, oddv = [0] * max_vals, [0] * max_vals
                    f1_perm.append(multiset_permutation_count(_even_f1_multiplicities(leader)))
                else:
                    f0c, f0v = [0] * max_vals, [0] * max_vals
                    f1c, f1v = [0] * max_vals, [0] * max_vals
                    oddc, oddv = counts_and_values(_odd_multiplicities(leader))
                    f1_perm.append(1)
                f0_counts.append(f0c)
                f1_counts.append(f1c)
                odd_counts.append(oddc)
                f0_values.append(f0v)
                f1_values.append(f1v)
                odd_values.append(oddv)

        golay_bits = []
        for codeword in self.golay_codewords:
            mask = 0
            for bit, value in enumerate(codeword):
                if value:
                    mask |= 1 << bit
            golay_bits.append(mask)

        self._dequantize_triton_meta = {
            "max_vals": max_vals,
            "n_classes": len(class_starts),
            "class_starts": torch.tensor(class_starts, dtype=torch.long, device=self.device),
            "parity": torch.tensor(parity, dtype=torch.int32, device=self.device),
            "perm_count": torch.tensor(perm_count, dtype=torch.long, device=self.device),
            "sign_count": torch.tensor(sign_count, dtype=torch.long, device=self.device),
            "f1_perm": torch.tensor(f1_perm, dtype=torch.long, device=self.device),
            "golay_start": torch.tensor(golay_start, dtype=torch.long, device=self.device),
            "golay_count": torch.tensor(golay_count, dtype=torch.long, device=self.device),
            "golay_indices": torch.tensor(golay_indices, dtype=torch.long, device=self.device),
            "golay_bits": torch.tensor(golay_bits, dtype=torch.int32, device=self.device),
            "f0_counts": torch.tensor(f0_counts, dtype=torch.long, device=self.device),
            "f1_counts": torch.tensor(f1_counts, dtype=torch.long, device=self.device),
            "odd_counts": torch.tensor(odd_counts, dtype=torch.long, device=self.device),
            "f0_values": torch.tensor(f0_values, dtype=torch.long, device=self.device),
            "f1_values": torch.tensor(f1_values, dtype=torch.long, device=self.device),
            "odd_values": torch.tensor(odd_values, dtype=torch.long, device=self.device),
        }
        return self._dequantize_triton_meta

    def _decode_metadata(self) -> dict:
        cached = getattr(self, "_decode_meta", None)
        if cached is not None:
            return cached

        self._decode_meta = {
            "shell_ends": torch.tensor(
                [self.cumulative_shell_counts[shell] for shell in range(2, self.max_shell + 1)],
                dtype=torch.long,
                device=self.device,
            ),
            "class_offsets": {
                shell: torch.tensor(self.class_offsets[shell], dtype=torch.long, device=self.device)
                for shell in range(2, self.max_shell + 1)
            },
        }
        flat_starts: list[int] = []
        flat_structures: list[ClassLocalStructure] = []
        for shell in range(2, self.max_shell + 1):
            shell_offset = self.shell_offsets[shell]
            for structure in self.local_structures[shell]:
                flat_starts.append(shell_offset + self.class_offsets[shell][structure.class_index])
                flat_structures.append(structure)
        self._decode_meta["class_starts"] = torch.tensor(
            flat_starts,
            dtype=torch.long,
            device=self.device,
        )
        self._decode_meta["flat_structures"] = tuple(flat_structures)
        return self._decode_meta

    def _rank_even_codewords_gpu(
        self,
        vectors: torch.Tensor,
        structure: ClassLocalStructure,
        golay_choice: int,
    ) -> torch.Tensor:
        golay_index = structure.golay_codeword_indices[golay_choice]
        codeword = self.golay_codewords[golay_index]
        f0_positions = [i for i, bit in enumerate(codeword) if bit == 0]
        f1_positions = [i for i, bit in enumerate(codeword) if bit == 1]
        f0_counts = _even_f0_multiplicities(structure.leader)
        f1_counts = _even_f1_multiplicities(structure.leader)
        f0_rank = self._rank_multiset_sequences_gpu(vectors[:, f0_positions].abs(), f0_counts)
        f1_rank = self._rank_multiset_sequences_gpu(vectors[:, f1_positions].abs(), f1_counts)
        f1_permutations = multiset_permutation_count(f1_counts)
        permutation_choice = f0_rank * f1_permutations + f1_rank
        sign_choice = self._rank_even_signs_gpu(vectors)
        return (
            (golay_choice * structure.sign_count + sign_choice)
            * structure.permutation_count
            + permutation_choice
        )

    def _rank_odd_codewords_gpu(
        self,
        vectors: torch.Tensor,
        structure: ClassLocalStructure,
        golay_choice: int,
    ) -> torch.Tensor:
        permutation_choice = self._rank_multiset_sequences_gpu(vectors.abs(), _odd_multiplicities(structure.leader))
        return golay_choice * structure.permutation_count + permutation_choice

    def _unrank_even_codewords_gpu(
        self,
        structure: ClassLocalStructure,
        golay_choice: int,
        sign_choice: torch.Tensor,
        permutation_choice: torch.Tensor,
    ) -> torch.Tensor:
        golay_index = structure.golay_codeword_indices[golay_choice]
        codeword = self.golay_codewords[golay_index]
        f0_positions = [i for i, bit in enumerate(codeword) if bit == 0]
        f1_positions = [i for i, bit in enumerate(codeword) if bit == 1]
        f0_counts = _even_f0_multiplicities(structure.leader)
        f1_counts = _even_f1_multiplicities(structure.leader)
        f1_permutations = multiset_permutation_count(f1_counts)
        f1_rank = permutation_choice.remainder(f1_permutations)
        f0_rank = torch.div(permutation_choice, f1_permutations, rounding_mode="floor")

        unsigned = torch.zeros((permutation_choice.shape[0], DIM), dtype=torch.long, device=self.device)
        if f0_positions:
            unsigned[:, f0_positions] = self._unrank_multiset_sequences_gpu(f0_counts, f0_rank)
        if f1_positions:
            unsigned[:, f1_positions] = self._unrank_multiset_sequences_gpu(f1_counts, f1_rank)
        return self._unrank_even_signs_gpu(unsigned, sign_choice)

    def _unrank_odd_codewords_gpu(
        self,
        structure: ClassLocalStructure,
        golay_choice: int,
        permutation_choice: torch.Tensor,
    ) -> torch.Tensor:
        golay_index = structure.golay_codeword_indices[golay_choice]
        codeword = torch.tensor(self.golay_codewords[golay_index], dtype=torch.long, device=self.device)
        magnitudes = self._unrank_multiset_sequences_gpu(_odd_multiplicities(structure.leader), permutation_choice)
        target_mod4 = torch.where(codeword == 1, 3, 1).unsqueeze(0)
        signs = torch.where(
            magnitudes.remainder(4) == target_mod4,
            torch.ones((), dtype=torch.long, device=self.device),
            -torch.ones((), dtype=torch.long, device=self.device),
        )
        return signs * magnitudes

    def _rank_multiset_sequences_gpu(
        self,
        sequences: torch.Tensor,
        multiplicities: dict[int, int],
    ) -> torch.Tensor:
        batch = sequences.shape[0]
        if sequences.shape[1] == 0:
            return torch.zeros((batch,), dtype=torch.long, device=self.device)

        values = tuple(sorted((value for value, count in multiplicities.items() if count), reverse=True))
        value_tensor = torch.tensor(values, dtype=torch.long, device=self.device)
        counts = torch.tensor([multiplicities[value] for value in values], dtype=torch.long, device=self.device)
        counts = counts.unsqueeze(0).expand(batch, -1).clone()
        rank = torch.zeros((batch,), dtype=torch.long, device=self.device)
        rows = torch.arange(batch, device=self.device)

        for pos in range(sequences.shape[1]):
            item = sequences[:, pos].to(torch.long)
            seen_item = torch.zeros((batch,), dtype=torch.bool, device=self.device)
            for value_index, value in enumerate(values):
                is_item = item == value
                add_mask = (~seen_item) & (~is_item) & (counts[:, value_index] > 0)
                if add_mask.any():
                    trial_counts = counts.clone()
                    trial_counts[:, value_index] -= 1
                    rank += torch.where(
                        add_mask,
                        self._multiset_permutation_count_gpu(trial_counts),
                        torch.zeros_like(rank),
                    )
                seen_item |= is_item

            choice = (item[:, None] == value_tensor[None, :]).to(torch.long).argmax(dim=1)
            counts[rows, choice] -= 1
        return rank

    def _unrank_multiset_sequences_gpu(
        self,
        multiplicities: dict[int, int],
        ranks: torch.Tensor,
    ) -> torch.Tensor:
        batch = ranks.shape[0]
        width = sum(multiplicities.values())
        if width == 0:
            return torch.empty((batch, 0), dtype=torch.long, device=self.device)

        values = tuple(sorted((value for value, count in multiplicities.items() if count), reverse=True))
        counts = torch.tensor([multiplicities[value] for value in values], dtype=torch.long, device=self.device)
        counts = counts.unsqueeze(0).expand(batch, -1).clone()
        ranks = ranks.clone()
        out = torch.zeros((batch, width), dtype=torch.long, device=self.device)

        for pos in range(width):
            chosen = torch.zeros((batch,), dtype=torch.bool, device=self.device)
            for value_index, value in enumerate(values):
                trial_counts = counts.clone()
                eligible = (~chosen) & (counts[:, value_index] > 0)
                trial_counts[:, value_index] -= 1
                block_count = self._multiset_permutation_count_gpu(trial_counts)
                take = eligible & (ranks < block_count)
                if take.any():
                    out[take, pos] = value
                    counts[take, value_index] -= 1
                    chosen |= take
                skip = eligible & (~take)
                ranks = torch.where(skip, ranks - block_count, ranks)
        return out

    def _multiset_permutation_count_gpu(self, counts: torch.Tensor) -> torch.Tensor:
        remaining = counts.sum(dim=1)
        out = torch.ones((counts.shape[0],), dtype=torch.long, device=self.device)
        for col in range(counts.shape[1]):
            chosen = counts[:, col]
            out *= self._binom[remaining, chosen]
            remaining -= chosen
        return out

    def _rank_even_signs_gpu(self, vectors: torch.Tensor) -> torch.Tensor:
        if (vectors.sum(dim=1).remainder(8) != 0).any():
            raise ValueError("even vector coordinate sum is not 0 mod 8")

        nonzero_count = int((vectors[0] != 0).sum().item())
        if nonzero_count == 0:
            return torch.zeros((vectors.shape[0],), dtype=torch.long, device=self.device)

        unsigned = vectors.abs()[vectors != 0].reshape(vectors.shape[0], nonzero_count).to(torch.long)
        target_bits = (vectors[vectors != 0].reshape(vectors.shape[0], nonzero_count) < 0)

        lower_counts: list[torch.Tensor] = []
        dp = torch.zeros((vectors.shape[0], 8), dtype=torch.long, device=self.device)
        dp[:, 0] = 1
        lower_counts.append(dp)
        residues = torch.arange(8, dtype=torch.long, device=self.device).unsqueeze(0)
        for pos in range(nonzero_count):
            mag = unsigned[:, pos].remainder(8).unsqueeze(1)
            next_dp = torch.zeros_like(dp)
            next_dp.scatter_add_(1, (residues + mag).remainder(8), dp)
            next_dp.scatter_add_(1, (residues - mag).remainder(8), dp)
            dp = next_dp
            lower_counts.append(dp)

        rank = torch.zeros((vectors.shape[0],), dtype=torch.long, device=self.device)
        partial_high = torch.zeros((vectors.shape[0],), dtype=torch.long, device=self.device)
        rows = torch.arange(vectors.shape[0], device=self.device)
        for pos in range(nonzero_count - 1, -1, -1):
            mag = unsigned[:, pos]
            bit_is_one = target_bits[:, pos]
            needed = (-(partial_high + mag)).remainder(8)
            rank += torch.where(
                bit_is_one,
                lower_counts[pos][rows, needed],
                torch.zeros_like(rank),
            )
            partial_high += torch.where(bit_is_one, -mag, mag)
        return rank

    def _unrank_even_signs_gpu(self, unsigned: torch.Tensor, sign_rank: torch.Tensor) -> torch.Tensor:
        nonzero_count = int((unsigned[0] != 0).sum().item())
        if nonzero_count == 0:
            return unsigned

        compact = unsigned[unsigned != 0].reshape(unsigned.shape[0], nonzero_count).to(torch.long)
        lower_counts: list[torch.Tensor] = []
        dp = torch.zeros((unsigned.shape[0], 8), dtype=torch.long, device=self.device)
        dp[:, 0] = 1
        lower_counts.append(dp)
        residues = torch.arange(8, dtype=torch.long, device=self.device).unsqueeze(0)
        for pos in range(nonzero_count):
            mag = compact[:, pos].remainder(8).unsqueeze(1)
            next_dp = torch.zeros_like(dp)
            next_dp.scatter_add_(1, (residues + mag).remainder(8), dp)
            next_dp.scatter_add_(1, (residues - mag).remainder(8), dp)
            dp = next_dp
            lower_counts.append(dp)

        bits = torch.zeros((unsigned.shape[0], nonzero_count), dtype=torch.bool, device=self.device)
        partial_high = torch.zeros((unsigned.shape[0],), dtype=torch.long, device=self.device)
        rows = torch.arange(unsigned.shape[0], device=self.device)
        rank = sign_rank.clone()
        for pos in range(nonzero_count - 1, -1, -1):
            mag = compact[:, pos]
            needed = (-(partial_high + mag)).remainder(8)
            zero_branch_count = lower_counts[pos][rows, needed]
            choose_one = rank >= zero_branch_count
            bits[:, pos] = choose_one
            rank = torch.where(choose_one, rank - zero_branch_count, rank)
            partial_high += torch.where(choose_one, -mag, mag)

        signed_compact = torch.where(bits, -compact, compact)
        signed = unsigned.clone()
        signed[unsigned != 0] = signed_compact.reshape(-1)
        return signed

    def quantize_exhaustive(self, vector: Sequence[float], max_candidates: int = 1_000_000) -> int:
        """Old reference path: scan every dequantized codeword."""
        values = tuple(float(value) for value in vector)
        if len(values) != DIM:
            raise ValueError(f"expected {DIM} coordinates, got {len(values)}")
        if self.total_count > max_candidates:
            raise ValueError(
                f"exhaustive quantize would scan {self.total_count} codewords; "
                "increase max_candidates to run this check"
            )

        best_index = 0
        best_distance = float("inf")
        for index in range(self.total_count):
            distance = squared_distance(values, self.dequantize_lattice(index))
            if distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

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
            cursor = 0
            offsets = []
            for row in rows:
                offsets.append(cursor)
                cursor += row.count
            out[shell] = tuple(offsets)
        return out

    def _build_local_structures(self) -> dict[int, tuple[ClassLocalStructure, ...]]:
        return {
            shell: tuple(
                build_local_structure(shell, i, row, self.golay_codewords)
                for i, row in enumerate(rows)
            )
            for shell, rows in self.classes_by_shell.items()
        }

    def _build_class_index(self) -> dict[int, dict[tuple[tuple[int, ...], str], int]]:
        out = {}
        for shell, rows in self.classes_by_shell.items():
            out[shell] = {(row.leader, row.parity): index for index, row in enumerate(rows)}
        return out

    def _validate(self) -> None:
        theta_counts = leech_shell_counts(self.max_shell)
        for shell, rows in self.classes_by_shell.items():
            table_count = sum(row.count for row in rows)
            if table_count != theta_counts[shell]:
                raise ValueError(
                    f"shell {shell} leaders sum to {table_count}, theta series gives {theta_counts[shell]}"
                )
            for structure in self.local_structures[shell]:
                if structure.count != structure.leader.count:
                    raise ValueError(
                        f"shell {shell}, class {structure.class_index} has local count "
                        f"{structure.count}, leader count {structure.leader.count}"
                    )


def build_local_structure(
    shell: int,
    class_index: int,
    leader: ClassLeader,
    golay_codewords: Sequence[tuple[int, ...]],
) -> ClassLocalStructure:
    if leader.parity == "even":
        weight = leader.even_weight
        golay_indices = tuple(i for i, c in enumerate(golay_codewords) if sum(c) == weight)
        f0_count = multiset_permutation_count(_even_f0_multiplicities(leader))
        f1_count = multiset_permutation_count(_even_f1_multiplicities(leader))
        if weight == 0 and sum(leader.leader) % 8 != 0:
            sign_count = 0
        else:
            sign_exponent = leader.nonzero_count if weight == 0 else leader.nonzero_count - 1
            sign_count = 1 << sign_exponent
        permutation_count = f0_count * f1_count
    elif leader.parity == "odd":
        golay_indices = tuple(range(len(golay_codewords)))
        sign_count = 1
        permutation_count = multiset_permutation_count(_odd_multiplicities(leader))
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


def generate_class_leaders(
    max_shell: int,
    cache_dir: str | Path | None = ".llvq_cache",
    use_cache: bool = True,
    verbose: bool = False,
) -> tuple[ClassLeader, ...]:
    """Generate class leaders from the integer shell equation."""
    leaders = []
    for shell in range(2, max_shell + 1):
        if use_cache:
            start = perf_counter()
            shell_leaders, cache_status = generate_shell_class_leaders_cached(shell, cache_dir=cache_dir)
            if verbose:
                print(
                    f"{cache_status} class leaders for shell m={shell}: "
                    f"{len(shell_leaders)} classes in {perf_counter() - start:.3f}s",
                    flush=True,
                )
        else:
            start = perf_counter()
            shell_leaders = generate_shell_class_leaders(shell)
            if verbose:
                print(
                    f"generated class leaders for shell m={shell}: "
                    f"{len(shell_leaders)} classes in {perf_counter() - start:.3f}s",
                    flush=True,
                )
        leaders.extend(shell_leaders)
    return tuple(leaders)


def generate_shell_class_leaders_cached(
    shell: int,
    cache_dir: str | Path | None = ".llvq_cache",
) -> tuple[tuple[ClassLeader, ...], str]:
    cache_key = None if cache_dir is None else str(Path(cache_dir))
    return _generate_shell_class_leaders_cached(shell, cache_key)


def _generate_shell_class_leaders_cached(
    shell: int,
    cache_dir: str | None,
) -> tuple[tuple[ClassLeader, ...], str]:
    cache_path = _class_leader_cache_path(cache_dir, shell)
    if cache_path is not None and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if (
                payload.get("version") == CLASS_LEADER_CACHE_VERSION
                and payload.get("shell") == shell
                and isinstance(payload.get("leaders"), tuple)
            ):
                return payload["leaders"], "loaded cached"
        except (OSError, pickle.PickleError, AttributeError, EOFError):
            pass

    leaders = generate_shell_class_leaders(shell)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "version": CLASS_LEADER_CACHE_VERSION,
                    "shell": shell,
                    "leaders": leaders,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
    return leaders, "generated"


def _class_leader_cache_path(cache_dir: str | None, shell: int) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"class_leaders_v{CLASS_LEADER_CACHE_VERSION}_shell{shell}.pkl"


def generate_shell_class_leaders(shell: int) -> tuple[ClassLeader, ...]:
    target_square_sum = 16 * shell
    leaders = []
    leaders.extend(generate_even_shell_leaders(shell, target_square_sum))
    leaders.extend(generate_odd_shell_leaders(shell, target_square_sum))
    return tuple(sorted(leaders, key=lambda row: (row.leader, 0 if row.parity == "even" else 1, row.count)))


def generate_even_shell_leaders(shell: int, target_square_sum: int) -> tuple[ClassLeader, ...]:
    max_abs = int(target_square_sum**0.5)
    magnitudes = tuple(value for value in range(0, max_abs + 1, 2))
    leaders = []
    for counts in _multiplicity_solutions(magnitudes, DIM, target_square_sum):
        leader = _leader_from_counts(shell, "even", counts, count=0)
        count = class_cardinality(leader)
        if count:
            leaders.append(_replace_count(leader, count))
    return tuple(leaders)


def generate_odd_shell_leaders(shell: int, target_square_sum: int) -> tuple[ClassLeader, ...]:
    max_abs = int(target_square_sum**0.5)
    magnitudes = tuple(value for value in range(1, max_abs + 1, 2))
    leaders = []
    for counts in _multiplicity_solutions(magnitudes, DIM, target_square_sum):
        leader = _leader_from_counts(shell, "odd", counts, count=0)
        count = class_cardinality(leader)
        if count:
            leaders.append(_replace_count(leader, count))
    return tuple(leaders)


def _multiplicity_solutions(
    magnitudes: Sequence[int],
    remaining_slots: int,
    remaining_square_sum: int,
) -> Iterable[dict[int, int]]:
    if not magnitudes:
        if remaining_slots == 0 and remaining_square_sum == 0:
            yield {}
        return

    magnitude = magnitudes[-1]
    square = magnitude * magnitude
    rest = magnitudes[:-1]
    max_count = remaining_slots if square == 0 else min(remaining_slots, remaining_square_sum // square)
    for count in range(max_count, -1, -1):
        next_slots = remaining_slots - count
        next_sum = remaining_square_sum - count * square
        for solution in _multiplicity_solutions(rest, next_slots, next_sum):
            if count:
                solution = dict(solution)
                solution[magnitude] = count
            yield solution


def class_cardinality(leader: ClassLeader) -> int:
    structure = build_local_structure(
        shell=leader.shell,
        class_index=0,
        leader=leader,
        golay_codewords=extended_golay_codewords(),
    )
    return structure.count


def _replace_count(leader: ClassLeader, count: int) -> ClassLeader:
    return _leader_from_counts(leader.shell, leader.parity, leader.multiplicities, count=count)


def _leader_from_counts(shell: int, parity: str, counts: dict[int, int], count: int) -> ClassLeader:
    fixed = {magnitude: counts.get(magnitude, 0) for magnitude in (8, 6, 5, 4, 3, 2, 1, 0)}
    extras = tuple(sorted(
        ((magnitude, value) for magnitude, value in counts.items() if magnitude not in fixed and value),
        reverse=True,
    ))
    return ClassLeader(
        shell=shell,
        parity=parity,
        count=count,
        mult_abs8=fixed[8],
        mult_abs6=fixed[6],
        mult_abs5=fixed[5],
        mult_abs4=fixed[4],
        mult_abs3=fixed[3],
        mult_abs2=fixed[2],
        mult_abs1=fixed[1],
        mult_abs0=fixed[0],
        extra_multiplicities=extras,
    )


def _even_f0_multiplicities(leader: ClassLeader) -> dict[int, int]:
    return {magnitude: count for magnitude, count in leader.multiplicities.items() if magnitude % 4 == 0}


def _even_f1_multiplicities(leader: ClassLeader) -> dict[int, int]:
    return {magnitude: count for magnitude, count in leader.multiplicities.items() if magnitude % 4 == 2}


def _odd_multiplicities(leader: ClassLeader) -> dict[int, int]:
    return {magnitude: count for magnitude, count in leader.multiplicities.items() if magnitude % 2 == 1}


def compare_with_reference(max_shell: int = 4) -> bool:
    generated = tuple(row for row in generate_class_leaders(max_shell) if row.shell <= max_shell)
    reference = tuple(
        sorted(
            (row for row in TABLE_CLASS_LEADERS_REFERENCE if row.shell <= max_shell),
            key=lambda row: (row.shell, row.leader, 0 if row.parity == "even" else 1, row.count),
        )
    )
    generated = tuple(
        sorted(generated, key=lambda row: (row.shell, row.leader, 0 if row.parity == "even" else 1, row.count))
    )
    return generated == reference


def solve_even_subclass(
    x: Sequence[float],
    leader: ClassLeader,
    golay_codeword: tuple[int, ...],
) -> tuple[float, tuple[int, ...]]:
    candidate = [0] * DIM
    f0_positions = tuple(i for i, bit in enumerate(golay_codeword) if bit == 0)
    f1_positions = tuple(i for i, bit in enumerate(golay_codeword) if bit == 1)
    f0_magnitudes = _ascending_multiset(_even_f0_multiplicities(leader))
    f1_magnitudes = _ascending_multiset(_even_f1_multiplicities(leader))

    ordered_f0 = sorted(f0_positions, key=lambda pos: (abs(x[pos]), pos))
    ordered_f1 = sorted(f1_positions, key=lambda pos: (abs(x[pos]), pos))

    for position, magnitude in zip(ordered_f0, f0_magnitudes, strict=True):
        candidate[position] = magnitude if x[position] >= 0 else -magnitude
    for position, magnitude in zip(ordered_f1, f1_magnitudes, strict=True):
        candidate[position] = magnitude if x[position] >= 0 else -magnitude

    if f1_magnitudes:
        required_negative_parity = (sum(leader.leader) % 8) // 4
        actual_negative_parity = sum(1 for position in f1_positions if candidate[position] < 0) % 2
        if actual_negative_parity != required_negative_parity:
            position_to_flip = ordered_f1[0]
            candidate[position_to_flip] = -candidate[position_to_flip]

    vector = tuple(candidate)
    return dot_product(x, vector), vector


def solve_odd_subclass(
    x: Sequence[float],
    leader: ClassLeader,
    golay_codeword: tuple[int, ...],
) -> tuple[float, tuple[int, ...]]:
    x_prime = tuple(-value if bit else value for value, bit in zip(x, golay_codeword, strict=True))
    ordered_positions = sorted(range(DIM), key=lambda pos: (x_prime[pos], pos))
    signed_leader = sorted(_odd_signed_leader_values(leader))

    candidate_prime = [0] * DIM
    for position, value in zip(ordered_positions, signed_leader, strict=True):
        candidate_prime[position] = value

    candidate = tuple(
        -value if bit else value
        for value, bit in zip(candidate_prime, golay_codeword, strict=True)
    )
    return dot_product(x, candidate), candidate


def _ascending_multiset(multiplicities: dict[int, int]) -> tuple[int, ...]:
    values = []
    for magnitude, count in sorted(multiplicities.items()):
        values.extend([magnitude] * count)
    return tuple(values)


def _odd_signed_leader_values(leader: ClassLeader) -> tuple[int, ...]:
    values = []
    for magnitude, count in _odd_multiplicities(leader).items():
        signed = magnitude if magnitude % 4 == 1 else -magnitude
        values.extend([signed] * count)
    return tuple(values)


def unrank_class_codeword(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    if leader.parity == "even":
        return unrank_even_codeword(leader, local)
    if leader.parity == "odd":
        return unrank_odd_codeword(leader, local)
    raise ValueError(f"unknown parity {leader.parity!r}")


def rank_class_codeword(
    vector: tuple[int, ...],
    structure: ClassLocalStructure,
    golay_index_by_codeword: dict[tuple[int, ...], int],
) -> int:
    if structure.leader.parity == "even":
        local = rank_even_codeword(vector, structure, golay_index_by_codeword)
    elif structure.leader.parity == "odd":
        local = rank_odd_codeword(vector, structure, golay_index_by_codeword)
    else:
        raise ValueError(f"unknown parity {structure.leader.parity!r}")

    return (
        (local.golay_choice * structure.sign_count + local.sign_choice)
        * structure.permutation_count
        + local.permutation_choice
    )


def rank_even_codeword(
    vector: tuple[int, ...],
    structure: ClassLocalStructure,
    golay_index_by_codeword: dict[tuple[int, ...], int],
) -> LocalDecomposition:
    codeword = tuple(1 if abs(value) % 4 == 2 else 0 for value in vector)
    golay_choice = _compatible_golay_choice(codeword, structure, golay_index_by_codeword)

    f0_values = tuple(abs(value) for value, bit in zip(vector, codeword, strict=True) if bit == 0)
    f1_values = tuple(abs(value) for value, bit in zip(vector, codeword, strict=True) if bit == 1)
    f0_counts = _even_f0_multiplicities(structure.leader)
    f1_counts = _even_f1_multiplicities(structure.leader)
    f0_rank = rank_multiset_sequence(f0_values, f0_counts)
    f1_rank = rank_multiset_sequence(f1_values, f1_counts)
    f1_permutations = multiset_permutation_count(f1_counts)
    permutation_choice = f0_rank * f1_permutations + f1_rank

    sign_choice = rank_even_signs(vector)
    if sign_choice >= structure.sign_count:
        raise ValueError("even sign pattern is outside the class sign range")

    return LocalDecomposition(
        golay_choice=golay_choice,
        golay_codeword=codeword,
        sign_choice=sign_choice,
        permutation_choice=permutation_choice,
    )


def rank_odd_codeword(
    vector: tuple[int, ...],
    structure: ClassLocalStructure,
    golay_index_by_codeword: dict[tuple[int, ...], int],
) -> LocalDecomposition:
    codeword = tuple(0 if value % 4 == 1 else 1 if value % 4 == 3 else -1 for value in vector)
    if any(bit < 0 for bit in codeword):
        raise ValueError("odd coordinates must be congruent to 1 or 3 mod 4")
    golay_choice = _compatible_golay_choice(codeword, structure, golay_index_by_codeword)
    magnitudes = tuple(abs(value) for value in vector)
    permutation_choice = rank_multiset_sequence(magnitudes, _odd_multiplicities(structure.leader))
    return LocalDecomposition(
        golay_choice=golay_choice,
        golay_codeword=codeword,
        sign_choice=0,
        permutation_choice=permutation_choice,
    )


def _compatible_golay_choice(
    codeword: tuple[int, ...],
    structure: ClassLocalStructure,
    golay_index_by_codeword: dict[tuple[int, ...], int],
) -> int:
    try:
        golay_index = golay_index_by_codeword[codeword]
    except KeyError as exc:
        raise ValueError("coordinate congruence pattern is not a Golay codeword") from exc
    try:
        return structure.golay_codeword_indices.index(golay_index)
    except ValueError as exc:
        raise ValueError("Golay codeword is not compatible with this class") from exc


def unrank_even_codeword(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    f0_positions = tuple(i for i, bit in enumerate(local.golay_codeword) if bit == 0)
    f1_positions = tuple(i for i, bit in enumerate(local.golay_codeword) if bit == 1)
    f0_counts = _even_f0_multiplicities(leader)
    f1_counts = _even_f1_multiplicities(leader)
    f1_permutations = multiset_permutation_count(f1_counts)
    f1_rank = local.permutation_choice % f1_permutations
    f0_rank = local.permutation_choice // f1_permutations

    unsigned = [0] * DIM
    for position, magnitude in zip(f0_positions, unrank_multiset_sequence(f0_counts, f0_rank), strict=True):
        unsigned[position] = magnitude
    for position, magnitude in zip(f1_positions, unrank_multiset_sequence(f1_counts, f1_rank), strict=True):
        unsigned[position] = magnitude
    return apply_even_signs(tuple(unsigned), local.sign_choice)


def unrank_odd_codeword(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    magnitudes = unrank_multiset_sequence(
        _odd_multiplicities(leader),
        local.permutation_choice,
    )
    values = []
    for magnitude, bit in zip(magnitudes, local.golay_codeword, strict=True):
        target_mod4 = 3 if bit else 1
        sign = 1 if magnitude % 4 == target_mod4 else -1
        values.append(sign * magnitude)
    return tuple(values)


def apply_even_signs(unsigned: tuple[int, ...], sign_rank: int) -> tuple[int, ...]:
    signable_positions = tuple(i for i, value in enumerate(unsigned) if value != 0)
    for mask in range(1 << len(signable_positions)):
        signed = list(unsigned)
        for bit, position in enumerate(signable_positions):
            if (mask >> bit) & 1:
                signed[position] = -signed[position]
        if sum(signed) % 8 != 0:
            continue
        if sign_rank == 0:
            return tuple(signed)
        sign_rank -= 1
    raise IndexError(sign_rank)


def rank_even_signs(vector: tuple[int, ...]) -> int:
    if sum(vector) % 8 != 0:
        raise ValueError("even vector coordinate sum is not 0 mod 8")

    unsigned = tuple(abs(value) for value in vector)
    signable_positions = tuple(i for i, value in enumerate(unsigned) if value != 0)
    target_mask = 0
    for bit, position in enumerate(signable_positions):
        if vector[position] < 0:
            target_mask |= 1 << bit

    rank = 0
    for mask in range(1 << len(signable_positions)):
        signed_sum = 0
        for bit, position in enumerate(signable_positions):
            sign = -1 if (mask >> bit) & 1 else 1
            signed_sum += sign * unsigned[position]
        if signed_sum % 8 != 0:
            continue
        if mask == target_mask:
            return rank
        rank += 1
    raise ValueError("even sign pattern is not valid for this class")


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
            if counts.get(value, 0) == 0:
                continue
            counts[value] -= 1
            branch_count = multiset_permutation_count(counts)
            if rank < branch_count:
                out.append(value)
                remaining -= 1
                break
            rank -= branch_count
            counts[value] += 1
        else:
            raise RuntimeError("failed to unrank multiset sequence")
    return tuple(out)


def rank_multiset_sequence(sequence: Sequence[int], multiplicities: dict[int, int]) -> int:
    counts = {value: count for value, count in multiplicities.items() if count}
    if len(sequence) != sum(counts.values()):
        raise ValueError("sequence length does not match multiplicities")

    rank = 0
    values = tuple(sorted(counts, reverse=True))
    for actual in sequence:
        if counts.get(actual, 0) == 0:
            raise ValueError(f"value {actual} is not available in this multiset")
        for candidate in values:
            if candidate == actual:
                break
            if counts.get(candidate, 0) == 0:
                continue
            counts[candidate] -= 1
            rank += multiset_permutation_count(counts)
            counts[candidate] += 1
        counts[actual] -= 1
    return rank


def multiset_permutation_count(multiplicities: dict[int, int]) -> int:
    total = sum(multiplicities.values())
    out = factorial(total)
    for count in multiplicities.values():
        out //= factorial(count)
    return out


def squared_distance(x: Sequence[float], y: Sequence[int]) -> float:
    return sum((xi - yi) * (xi - yi) for xi, yi in zip(x, y, strict=True))


def dot_product(x: Sequence[float], y: Sequence[int]) -> float:
    return sum(xi * yi for xi, yi in zip(x, y, strict=True))


def format_float_vector(vector: Sequence[float] | None, digits: int = 4) -> tuple[float, ...]:
    if vector is None:
        return ()
    return tuple(round(value, digits) for value in vector)


@lru_cache(maxsize=1)
def extended_golay_codewords() -> tuple[tuple[int, ...], ...]:
    """
    Extended binary Golay code G24 in lexicographic order.

    This uses the cyclic perfect Golay [23,12,7] generator
    g(x)=x^11+x^9+x^7+x^6+x^5+x+1, then appends the parity bit.
    """
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


def leech_shell_counts(max_shell: int) -> dict[int, int]:
    e4 = [0] * (max_shell + 1)
    e4[0] = 1
    for n in range(1, max_shell + 1):
        e4[n] = 240 * sigma_power(n, 3)
    e4_cubed = poly_pow_trunc(e4, 3, max_shell)
    delta = delta_coefficients(max_shell)
    return {m: e4_cubed[m] - 720 * delta[m] for m in range(2, max_shell + 1)}


def sigma_power(n: int, power: int) -> int:
    total = 0
    divisor = 1
    while divisor * divisor <= n:
        if n % divisor == 0:
            total += divisor**power
            other = n // divisor
            if other != divisor:
                total += other**power
        divisor += 1
    return total


def poly_mul_trunc(a: Sequence[int], b: Sequence[int], degree: int) -> list[int]:
    out = [0] * (degree + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b[: degree + 1 - i]):
            if bj:
                out[i + j] += ai * bj
    return out


def poly_pow_trunc(base: Sequence[int], exponent: int, degree: int) -> list[int]:
    out = [0] * (degree + 1)
    out[0] = 1
    power = list(base)
    while exponent:
        if exponent & 1:
            out = poly_mul_trunc(out, power, degree)
        exponent >>= 1
        if exponent:
            power = poly_mul_trunc(power, power, degree)
    return out


def delta_coefficients(degree: int) -> list[int]:
    coeffs = [0] * (degree + 1)
    coeffs[0] = 1
    binom24 = [factorial(24) // (factorial(k) * factorial(24 - k)) for k in range(25)]
    for n in range(1, degree + 1):
        factor = [0] * (degree + 1)
        for k, choose in enumerate(binom24):
            exponent = n * k
            if exponent > degree:
                break
            factor[exponent] = (-1) ** k * choose
        coeffs = poly_mul_trunc(coeffs, factor, degree)

    shifted = [0] * (degree + 1)
    for i in range(degree):
        shifted[i + 1] = coeffs[i]
    return shifted


def demo(
    max_shell: int = 2,
    samples: int = 1,
    seed: int = 0,
    max_candidates: int = 1_000_000,
) -> None:
    if samples < 1:
        raise ValueError("samples must be at least 1")

    rng = Random(seed)
    sample_vectors = [tuple(rng.gauss(0.0, 1.0) for _ in range(DIM)) for _ in range(samples)]
    print(f"samples: {samples}")
    print(f"input vector[0]: {format_float_vector(sample_vectors[0])}")

    db = LeechLatticeVectorQuantizerGpu(max_shell=max_shell)
    x = torch.tensor(sample_vectors, dtype=db.dtype, device=db.device)
    quantized_indices = db.quantize(x)
    quantized_vectors = db.dequantize(quantized_indices)
    total_squared_error = (x - quantized_vectors).square().sum().item()
    first_index = int(quantized_indices[0].item())
    first_exhaustive_index: int | None = None
    exhaustive_matches = True
    exhaustive_checked = db.total_count <= max_candidates

    if exhaustive_checked:
        for sample_index, vector in enumerate(sample_vectors):
            exhaustive_index = db.quantize_exhaustive(vector, max_candidates=max_candidates)
            quantized_index = int(quantized_indices[sample_index].item())
            exhaustive_matches = exhaustive_matches and (quantized_index == exhaustive_index)
            if sample_index == 0:
                first_exhaustive_index = exhaustive_index

    mse = total_squared_error / (samples * DIM)
    ranked = db.unrank(first_index)
    local = db.decompose_local_index(ranked)

    print(f"shells: 2..{max_shell}")
    print(f"total codewords: {db.total_count}")
    print(f"shape bits: {db.shape_bits}")
    print(f"shape bits/dim: {db.shape_bits / DIM:.6f}")
    if max_shell <= 4:
        print(f"generated leaders match reference table: {compare_with_reference(max_shell)}")
    print(f"quantized index[0]:   {first_index}")
    if exhaustive_checked:
        print(f"exhaustive index[0]:  {first_exhaustive_index}")
        print(f"nn matches exhaustive:{exhaustive_matches}")
    else:
        print(f"exhaustive check:     skipped; {db.total_count} codewords exceeds max_candidates")
    print(
        "quantized address[0]: "
        f"shell={ranked.shell}, class={ranked.class_index}, "
        f"class_local={ranked.class_local_index}, golay={local.golay_choice}, "
        f"sign={local.sign_choice}, perm={local.permutation_choice}"
    )
    print(f"integer codeword[0]:  {db.dequantize_lattice(first_index)}")
    print(f"dequantized vector[0]:{format_float_vector(db.dequantize(first_index))}")
    print(f"mse: {mse:.6f}")


def check_demo(max_shell: int = 4, indices: Iterable[int] = (0, 1103, 1104, 196559, 196560)) -> None:
    db = LeechLatticeVectorQuantizerGpu(max_shell=max_shell)
    print(f"shells: 2..{max_shell}")
    print(f"total codewords: {db.total_count}")
    print(f"shape bits: {db.shape_bits}")
    if max_shell <= 4:
        print(f"generated leaders match reference table: {compare_with_reference(max_shell)}")
    for shell in range(2, max_shell + 1):
        print(f"m={shell}: n(m)={db.shell_counts[shell]}, N(m)={db.cumulative_shell_counts[shell]}")
        for i, structure in enumerate(db.local_structures[shell]):
            leader = structure.leader
            print(
                f"  class={i} parity={leader.parity} leader={leader.leader} "
                f"count={leader.count} golay={len(structure.golay_codeword_indices)} "
                f"signs={structure.sign_count} perms={structure.permutation_count}"
            )

    for index in indices:
        ranked = db.unrank(index)
        local = db.decompose_local_index(ranked)
        word = db.dequantize(index)
        quantized_index = db.encode_codeword(word)
        print(
            f"index={index}: shell={ranked.shell} class={ranked.class_index} "
            f"local={ranked.class_local_index} golay={local.golay_choice} "
            f"sign={local.sign_choice} perm={local.permutation_choice} "
            f"dequantize={word} quantize={quantized_index}"
        )


def benchmark(
    max_shell: int = 4,
    samples: int = 64,
    seed: int = 0,
    subclass_chunk_size: int = 512,
    device: str | None = None,
    include_gpu_loop: bool = True,
) -> None:
    from impl.leech_lattice_vector_quantizer import LeechLatticeVectorQuantizer

    if samples < 1:
        raise ValueError("samples must be at least 1")

    rng = Random(seed)
    sample_vectors = [tuple(rng.gauss(0.0, 1.0) for _ in range(DIM)) for _ in range(samples)]

    print(f"max_shell: {max_shell}")
    print(f"samples: {samples}")
    print(f"subclass_chunk_size: {subclass_chunk_size}")

    cpu_q = LeechLatticeVectorQuantizer(max_shell=max_shell)
    start = perf_counter()
    cpu_indices = [cpu_q.quantize(vector) for vector in sample_vectors]
    cpu_seconds = perf_counter() - start
    print(
        f"cpu original structured: {cpu_seconds:.6f}s total, "
        f"{cpu_seconds / samples * 1000.0:.3f} ms/vector"
    )

    gpu_q = LeechLatticeVectorQuantizerGpu(max_shell=max_shell, device=device)
    print(f"gpu device: {gpu_q.device}")
    x = torch.tensor(sample_vectors, dtype=gpu_q.dtype, device=gpu_q.device)

    if gpu_q.device.type == "cuda":
        torch.cuda.synchronize(gpu_q.device)
    start = perf_counter()
    optimized_indices = gpu_q.quantize_optimized(x, subclass_chunk_size=subclass_chunk_size)
    if gpu_q.device.type == "cuda":
        torch.cuda.synchronize(gpu_q.device)
    optimized_seconds = perf_counter() - start
    optimized_list = optimized_indices.detach().cpu().tolist()
    print(
        f"gpu optimized chunked:   {optimized_seconds:.6f}s total, "
        f"{optimized_seconds / samples * 1000.0:.3f} ms/vector"
    )
    print(f"optimized matches cpu:  {optimized_list == cpu_indices}")

    if gpu_q.device.type == "cuda":
        # Warm up/compile Triton outside the timing window.
        _ = gpu_q.quantize_triton(x[:1])
        torch.cuda.synchronize(gpu_q.device)
        start = perf_counter()
        triton_indices = gpu_q.quantize_triton(x)
        torch.cuda.synchronize(gpu_q.device)
        triton_seconds = perf_counter() - start
        triton_list = triton_indices.detach().cpu().tolist()
        print(
            f"gpu triton score-only:  {triton_seconds:.6f}s total, "
            f"{triton_seconds / samples * 1000.0:.3f} ms/vector"
        )
        print(f"triton matches cpu:     {triton_list == cpu_indices}")
        if triton_seconds > 0:
            print(f"triton speedup vs cpu:  {cpu_seconds / triton_seconds:.2f}x")

        torch.cuda.synchronize(gpu_q.device)
        start = perf_counter()
        _ = gpu_q.dequantize(triton_indices)
        torch.cuda.synchronize(gpu_q.device)
        dequantize_seconds = perf_counter() - start
        print(
            f"raw dequantize:         {dequantize_seconds:.6f}s total, "
            f"{dequantize_seconds / samples * 1000.0:.3f} ms/vector"
        )

    if include_gpu_loop:
        if gpu_q.device.type == "cuda":
            # Warm up/compile the fused CUDA quantizer outside the timing window.
            _ = gpu_q.quantize_cuda(x[:1])
            torch.cuda.synchronize(gpu_q.device)
        start = perf_counter()
        loop_indices = gpu_q.quantize(x)
        if gpu_q.device.type == "cuda":
            torch.cuda.synchronize(gpu_q.device)
        loop_seconds = perf_counter() - start
        loop_list = loop_indices.detach().cpu().tolist()
        print(
            f"gpu cuda fused:         {loop_seconds:.6f}s total, "
            f"{loop_seconds / samples * 1000.0:.3f} ms/vector"
        )
        print(f"cuda fused matches cpu: {loop_list == cpu_indices}")

    if optimized_seconds > 0:
        print(f"speedup vs cpu:         {cpu_seconds / optimized_seconds:.2f}x")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Implicit Leech shell/class/local codeword database")
    parser.add_argument("--max-shell", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--subclass-chunk-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-gpu-loop", action="store_true")
    parser.add_argument("--indices", type=str, default="0,1103,1104,196559,196560")
    args = parser.parse_args()
    if args.check:
        indices = [int(part) for part in args.indices.split(",") if part.strip()]
        check_demo(max_shell=args.max_shell, indices=indices)
    elif args.benchmark:
        benchmark(
            max_shell=args.max_shell,
            samples=args.samples,
            seed=args.seed,
            subclass_chunk_size=args.subclass_chunk_size,
            device=args.device,
            include_gpu_loop=not args.no_gpu_loop,
        )
    else:
        demo(
            max_shell=args.max_shell,
            samples=args.samples,
            seed=args.seed,
            max_candidates=args.max_candidates,
        )


if __name__ == "__main__":
    main()
