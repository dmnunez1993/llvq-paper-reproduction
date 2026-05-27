from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from time import perf_counter

import torch

from llvq_adoul_barth_fast_segments import FastSegmentShellDatabase
from llvq_adoul_barth_triton_candidates import LLVQAdoulBarthTritonCandidates
from llvq_implicit import DIM


@dataclass
class ShapeGainResult:
    indices: torch.Tensor
    shape_indices: torch.Tensor
    gain_indices: torch.Tensor
    raw_shapes: torch.Tensor
    unit_shapes: torch.Tensor
    gains: torch.Tensor
    recon: torch.Tensor
    angular_scores: torch.Tensor


class ChiGainQuantizer:
    """Small Lloyd scalar quantizer for gains distributed like scale * sqrt(chi2_24)."""

    def __init__(
        self,
        bits: int = 1,
        dim: int = DIM,
        scale: float = 1.0,
        samples: int = 262_144,
        iters: int = 30,
        seed: int = 0,
        device: str | torch.device = "cuda",
    ):
        if bits < 0:
            raise ValueError("bits must be non-negative")
        self.bits = bits
        self.dim = dim
        self.scale = float(scale)
        self.levels = self._build_levels(samples, iters, seed).to(device=device, dtype=torch.float32)

    @property
    def size(self) -> int:
        return int(self.levels.numel())

    def quantize(self, gains: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gains = gains.to(device=self.levels.device, dtype=self.levels.dtype)
        distances = (gains[..., None] - self.levels[None, :]).abs()
        indices = distances.argmin(dim=-1)
        return indices, self.levels[indices]

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(device=self.levels.device, dtype=torch.long)
        return self.levels[indices]

    def _build_levels(self, samples: int, iters: int, seed: int) -> torch.Tensor:
        levels_count = 1 << self.bits
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sample = torch.randn(samples, self.dim, generator=generator).square().sum(dim=1).sqrt()
        sample = sample.sort().values * self.scale

        if levels_count == 1:
            return sample.mean().reshape(1)

        q = (torch.arange(levels_count, dtype=torch.float32) + 0.5) / levels_count
        levels = torch.quantile(sample, q)

        for _ in range(iters):
            boundaries = (levels[:-1] + levels[1:]) * 0.5
            bins = torch.bucketize(sample, boundaries)
            new_levels = levels.clone()
            for i in range(levels_count):
                values = sample[bins == i]
                if values.numel() > 0:
                    new_levels[i] = values.mean()
            if torch.allclose(new_levels, levels):
                break
            levels = new_levels

        return levels


class ShapeGainLLVQ:
    """Shape-gain wrapper around angular LLVQ shape search."""

    def __init__(
        self,
        shell_db: FastSegmentShellDatabase,
        gain_bits: int = 1,
        gain_scale: float = 1.0,
        block_n: int = 128,
        gain_samples: int = 262_144,
    ):
        self.db = shell_db
        self.shape_quantizer = LLVQAdoulBarthTritonCandidates(shell_db, block_n=block_n)
        self.gain_quantizer = ChiGainQuantizer(
            bits=gain_bits,
            scale=gain_scale,
            samples=gain_samples,
            device=shell_db.device,
        )

    def quantize(self, x: torch.Tensor, optimal_gain: bool = True) -> ShapeGainResult:
        x = torch.as_tensor(x, dtype=self.db.dtype, device=self.db.device)
        single = x.ndim == 1
        x = x.reshape(-1, DIM)

        shape_indices, raw_shapes, angular_scores = self.shape_quantizer.quantize(
            x,
            mode="angular",
            return_vectors=True,
        )
        raw_shapes = raw_shapes.reshape(-1, DIM)
        shape_norms = raw_shapes.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        unit_shapes = raw_shapes / shape_norms

        if optimal_gain:
            target_gains = (x * unit_shapes).sum(dim=1).clamp_min(0.0)
        else:
            target_gains = x.norm(dim=1)

        gain_indices, gains = self.gain_quantizer.quantize(target_gains)
        recon = gains[:, None] * unit_shapes
        indices = shape_indices.to(torch.long) * self.gain_quantizer.size + gain_indices.to(torch.long)

        if single:
            indices = indices[0]
            shape_indices = shape_indices[0]
            gain_indices = gain_indices[0]
            raw_shapes = raw_shapes[0]
            unit_shapes = unit_shapes[0]
            gains = gains[0]
            recon = recon[0]
            angular_scores = angular_scores[0]

        return ShapeGainResult(
            indices=indices,
            shape_indices=shape_indices,
            gain_indices=gain_indices,
            raw_shapes=raw_shapes,
            unit_shapes=unit_shapes,
            gains=gains,
            recon=recon,
            angular_scores=angular_scores,
        )

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.db.device)
        shape_indices = indices // self.gain_quantizer.size
        gain_indices = indices % self.gain_quantizer.size
        raw_shapes = self.shape_quantizer.dequantize(shape_indices).reshape(-1, DIM)
        unit_shapes = raw_shapes / raw_shapes.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        gains = self.gain_quantizer.dequantize(gain_indices.reshape(-1))
        recon = gains[:, None] * unit_shapes
        return recon[0] if indices.ndim == 0 else recon.reshape(*indices.shape, DIM)


def demo(
    M: int = 12,
    max_coord: int = 5,
    gain_bits: int = 1,
    batch_vectors: int = 2,
    input_scale: float = 3.0,
    gain_scale: float | None = None,
    block_n: int = 128,
    gain_samples: int = 262_144,
    device: str | None = "cuda",
    cache_dir: str | None = ".llvq_cache",
    no_cache: bool = False,
    verbose: bool = True,
    build_progress_every: int = 512,
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

    x = torch.randn(batch_vectors, DIM, device=db.device) * input_scale
    if gain_scale is None:
        gain_scale = float(x.square().mean().sqrt().item())

    quantizer = ShapeGainLLVQ(
        db,
        gain_bits=gain_bits,
        gain_scale=gain_scale,
        block_n=block_n,
        gain_samples=gain_samples,
    )

    if db.device.type == "cuda":
        torch.cuda.synchronize(db.device)
    quantize_start = perf_counter()
    result = quantizer.quantize(x, optimal_gain=True)
    if db.device.type == "cuda":
        torch.cuda.synchronize(db.device)
    quantize_seconds = perf_counter() - quantize_start

    if db.device.type == "cuda":
        torch.cuda.synchronize(db.device)
    dequantize_start = perf_counter()
    timed_recon = quantizer.dequantize(result.indices)
    if db.device.type == "cuda":
        torch.cuda.synchronize(db.device)
    dequantize_seconds = perf_counter() - dequantize_start

    raw_diff = x - result.raw_shapes
    optimal_unquantized_gain = (x * result.unit_shapes).sum(dim=1, keepdim=True).clamp_min(0.0)
    optimal_recon = optimal_unquantized_gain * result.unit_shapes
    optimal_diff = x - optimal_recon
    sg_diff = x - result.recon

    shape_bits = ceil(log2(db.total_count)) if db.total_count > 1 else 1
    total_bits = shape_bits + gain_bits
    total_bytes = ceil(total_bits * batch_vectors / 8)

    print("device:", db.device)
    print("shell sizes:", db.shell_sizes())
    print("total implicit shape vectors:", db.total_count)
    print("structural segments:", len(db.segments))
    print(f"build time: {build_seconds:.6f}s")
    print(f"quantize time: {quantize_seconds:.6f}s")
    print(f"dequantize time: {dequantize_seconds:.6f}s")
    print("timed dequantize matches:", torch.allclose(result.recon, timed_recon))
    print(f"shape bits/vector: {shape_bits}")
    print(f"gain bits/vector: {gain_bits}")
    print(f"total bits/vector: {total_bits} ({total_bits / DIM:.4f} bits/dim)")
    print(f"packed bytes for batch: {total_bytes}")
    print("gain levels:", quantizer.gain_quantizer.levels)
    print("indices:", result.indices)
    print("shape indices:", result.shape_indices)
    print("gain indices:", result.gain_indices)
    print("angular scores:", result.angular_scores)
    print("gains:", result.gains)
    print("x:", x)
    print("raw shape:", result.raw_shapes)
    print("shape-gain recon:", result.recon)
    print("raw shape mse:", raw_diff.square().mean())
    print("optimal unquantized gain mse:", optimal_diff.square().mean())
    print("shape-gain mse:", sg_diff.square().mean())
    print("shape-gain per-vector l2 error:", sg_diff.norm(dim=1))


def _argparse_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LLVQ shape-gain demo")
    parser.add_argument("--M", type=int, default=12)
    parser.add_argument("--max_coord", type=int, default=5)
    parser.add_argument("--gain_bits", type=int, default=1)
    parser.add_argument("--batch_vectors", type=int, default=2)
    parser.add_argument("--input_scale", type=float, default=3.0)
    parser.add_argument("--gain_scale", type=float, default=None)
    parser.add_argument("--block_n", type=int, default=128)
    parser.add_argument("--gain_samples", type=int, default=262_144)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=".llvq_cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--build_progress_every", type=int, default=512)
    args = parser.parse_args()

    demo(
        M=args.M,
        max_coord=args.max_coord,
        gain_bits=args.gain_bits,
        batch_vectors=args.batch_vectors,
        input_scale=args.input_scale,
        gain_scale=args.gain_scale,
        block_n=args.block_n,
        gain_samples=args.gain_samples,
        device=args.device,
        cache_dir=args.cache_dir,
        no_cache=args.no_cache,
        verbose=not args.quiet,
        build_progress_every=args.build_progress_every,
    )


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        _argparse_main()
    else:
        fire.Fire(demo)
