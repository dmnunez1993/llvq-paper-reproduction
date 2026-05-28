from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from time import perf_counter

import torch

from llvq_adoul_barth_fast_segments import FastSegmentShellDatabase
from llvq_adoul_barth_triton_candidates import LLVQAdoulBarthTritonCandidates
from llvq_shape_gain import ChiGainQuantizer
from llvq_true_leech import true_leech_rate
from llvq_implicit import DIM


@dataclass
class QuantizedBlocks:
    indices: torch.Tensor
    shape_indices: torch.Tensor
    gain_indices: torch.Tensor
    recon: torch.Tensor
    raw_shapes: torch.Tensor
    unit_shapes: torch.Tensor
    gains: torch.Tensor
    angular_scores: torch.Tensor


class PaperLikeLLVQ:
    """
    Paper-like LLVQ shape-gain quantizer.

    This file uses the true Leech shell cardinalities/rates from
    `llvq_true_leech.py`, so `M=12, gain_bits=1` reports the paper's
    48 bits/vector = 2 bits/dim budget.

    Important: the shape search is still the current approximate prototype
    (`LLVQAdoulBarthTritonCandidates`), not the full paper class-leader
    implementation. This file is a working quantizer with paper-compatible rate
    accounting, not a complete reproduction.
    """

    def __init__(
        self,
        M: int = 12,
        gain_bits: int = 1,
        max_coord: int | None = None,
        device: str | torch.device | None = "cuda",
        cache_dir: str | None = ".llvq_cache",
        block_n: int = 128,
        gain_scale: float = 1.0,
        gain_samples: int = 262_144,
        verbose: bool = False,
    ):
        self.M = M
        self.gain_bits = gain_bits
        self.max_coord = max_coord if max_coord is not None else ceil(sqrt(2 * M))
        self.true_rate = true_leech_rate(M, gain_bits=gain_bits)

        self.db = FastSegmentShellDatabase(device=device)
        self.db.build(
            M=M,
            max_coord=self.max_coord,
            verbose=verbose,
            cache_dir=cache_dir,
        )
        self.shape_quantizer = LLVQAdoulBarthTritonCandidates(self.db, block_n=block_n)
        self.gain_quantizer = ChiGainQuantizer(
            bits=gain_bits,
            scale=gain_scale,
            samples=gain_samples,
            device=self.db.device,
        )

    def quantize(self, x: torch.Tensor, optimal_gain: bool = True) -> QuantizedBlocks:
        x = torch.as_tensor(x, dtype=self.db.dtype, device=self.db.device)
        original_shape = x.shape
        if x.shape[-1] != DIM:
            raise ValueError(f"expected last dimension {DIM}, got {x.shape[-1]}")
        x = x.reshape(-1, DIM)

        shape_indices, raw_shapes, angular_scores = self.shape_quantizer.quantize(
            x,
            mode="angular",
            return_vectors=True,
        )
        raw_shapes = raw_shapes.reshape(-1, DIM)
        unit_shapes = raw_shapes / raw_shapes.norm(dim=1, keepdim=True).clamp_min(1.0e-12)

        if optimal_gain:
            target_gains = (x * unit_shapes).sum(dim=1).clamp_min(0.0)
        else:
            target_gains = x.norm(dim=1)

        gain_indices, gains = self.gain_quantizer.quantize(target_gains)
        recon = gains[:, None] * unit_shapes
        indices = shape_indices.to(torch.long) * self.gain_quantizer.size + gain_indices.to(torch.long)

        return QuantizedBlocks(
            indices=indices.reshape(original_shape[:-1]),
            shape_indices=shape_indices.reshape(original_shape[:-1]),
            gain_indices=gain_indices.reshape(original_shape[:-1]),
            recon=recon.reshape(original_shape),
            raw_shapes=raw_shapes.reshape(original_shape),
            unit_shapes=unit_shapes.reshape(original_shape),
            gains=gains.reshape(original_shape[:-1]),
            angular_scores=angular_scores.reshape(original_shape[:-1]),
        )

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.db.device)
        shape_indices = indices // self.gain_quantizer.size
        gain_indices = indices % self.gain_quantizer.size
        raw_shapes = self.shape_quantizer.dequantize(shape_indices).reshape(-1, DIM)
        unit_shapes = raw_shapes / raw_shapes.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        gains = self.gain_quantizer.dequantize(gain_indices.reshape(-1))
        recon = gains[:, None] * unit_shapes
        return recon.reshape(*indices.shape, DIM)


def demo(
    M: int = 12,
    gain_bits: int = 1,
    max_coord: int | None = None,
    batch_vectors: int = 2,
    input_scale: float = 3.0,
    gain_scale: float | None = None,
    gain_samples: int = 262_144,
    device: str | None = "cuda",
    cache_dir: str | None = ".llvq_cache",
    block_n: int = 128,
    quiet: bool = False,
) -> None:
    x = torch.randn(batch_vectors, DIM, device=device) * input_scale
    if gain_scale is None:
        gain_scale = float(x.square().mean().sqrt().item())

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = perf_counter()
    quantizer = PaperLikeLLVQ(
        M=M,
        gain_bits=gain_bits,
        max_coord=max_coord,
        device=device,
        cache_dir=cache_dir,
        block_n=block_n,
        gain_scale=gain_scale,
        gain_samples=gain_samples,
        verbose=not quiet,
    )
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    init_seconds = perf_counter() - start

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    quantize_start = perf_counter()
    q = quantizer.quantize(x, optimal_gain=True)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    quantize_seconds = perf_counter() - quantize_start

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    dequantize_start = perf_counter()
    timed_recon = quantizer.dequantize(q.indices)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    dequantize_seconds = perf_counter() - dequantize_start

    raw_mse = (x - q.raw_shapes).square().mean()
    sg_mse = (x - q.recon).square().mean()
    l2 = (x - q.recon).norm(dim=1)

    print("device:", quantizer.db.device)
    print("M:", M)
    print("max_coord used by prototype:", quantizer.max_coord)
    print("true Leech cumulative count:", quantizer.true_rate.cumulative_count)
    print("true shape bits/vector:", quantizer.true_rate.shape_bits_per_vector)
    print("gain bits/vector:", gain_bits)
    print("true total bits/vector:", quantizer.true_rate.total_bits_per_vector)
    print(f"true total bits/dim: {quantizer.true_rate.total_bits_per_dim:.6f}")
    print("prototype represented vectors:", quantizer.db.total_count)
    print("prototype segments:", len(quantizer.db.segments))
    print(f"init/build time: {init_seconds:.6f}s")
    print(f"quantize time: {quantize_seconds:.6f}s")
    print(f"dequantize time: {dequantize_seconds:.6f}s")
    print("timed dequantize matches:", torch.allclose(q.recon, timed_recon))
    print("gain levels:", quantizer.gain_quantizer.levels)
    print("indices:", q.indices)
    print("shape indices:", q.shape_indices)
    print("gain indices:", q.gain_indices)
    print("angular scores:", q.angular_scores)
    print("gains:", q.gains)
    print("raw shape mse:", raw_mse)
    print("shape-gain mse:", sg_mse)
    print("shape-gain per-vector l2 error:", l2)
    print("x:", x)
    print("raw shape:", q.raw_shapes)
    print("recon:", q.recon)


def _argparse_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Paper-rate LLVQ shape-gain quantizer prototype")
    parser.add_argument("--M", type=int, default=12)
    parser.add_argument("--gain_bits", type=int, default=1)
    parser.add_argument("--max_coord", type=int, default=None)
    parser.add_argument("--batch_vectors", type=int, default=2)
    parser.add_argument("--input_scale", type=float, default=3.0)
    parser.add_argument("--gain_scale", type=float, default=None)
    parser.add_argument("--gain_samples", type=int, default=262_144)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=".llvq_cache")
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    demo(
        M=args.M,
        gain_bits=args.gain_bits,
        max_coord=args.max_coord,
        batch_vectors=args.batch_vectors,
        input_scale=args.input_scale,
        gain_scale=args.gain_scale,
        gain_samples=args.gain_samples,
        device=args.device,
        cache_dir=args.cache_dir,
        block_n=args.block_n,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        _argparse_main()
    else:
        fire.Fire(demo)
