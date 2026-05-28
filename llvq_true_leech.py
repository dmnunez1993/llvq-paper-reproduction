from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2


def sigma_power(n: int, power: int) -> int:
    """Return sum_{d|n} d^power."""
    if n < 1:
        return 0
    total = 0
    d = 1
    while d * d <= n:
        if n % d == 0:
            total += d**power
            other = n // d
            if other != d:
                total += other**power
        d += 1
    return total


def poly_mul_trunc(a: list[int], b: list[int], degree: int) -> list[int]:
    out = [0] * (degree + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b[: degree + 1 - i]):
            if bj:
                out[i + j] += ai * bj
    return out


def poly_pow_trunc(base: list[int], exponent: int, degree: int) -> list[int]:
    out = [0] * (degree + 1)
    out[0] = 1
    power = base[:]
    exp = exponent
    while exp:
        if exp & 1:
            out = poly_mul_trunc(out, power, degree)
        exp >>= 1
        if exp:
            power = poly_mul_trunc(power, power, degree)
    return out


def delta_coefficients(degree: int) -> list[int]:
    """
    Coefficients of Ramanujan Delta:

        Delta(q) = q * prod_{n>=1} (1 - q^n)^24.
    """
    coeffs = [0] * (degree + 1)
    coeffs[0] = 1
    for n in range(1, degree + 1):
        factor = [0] * (degree + 1)
        for k in range(0, 25):
            exp = n * k
            if exp > degree:
                break
            factor[exp] = (-1) ** k * _binom_24(k)
        coeffs = poly_mul_trunc(coeffs, factor, degree)

    shifted = [0] * (degree + 1)
    for i in range(degree):
        shifted[i + 1] = coeffs[i]
    return shifted


def _binom_24(k: int) -> int:
    values = [
        1,
        24,
        276,
        2024,
        10626,
        42504,
        134596,
        346104,
        735471,
        1307504,
        1961256,
        2496144,
        2704156,
        2496144,
        1961256,
        1307504,
        735471,
        346104,
        134596,
        42504,
        10626,
        2024,
        276,
        24,
        1,
    ]
    return values[k]


def leech_shell_counts(M: int) -> dict[int, int]:
    """
    Exact Leech lattice shell counts from the theta series.

    With q indexed by m = ||x||^2 / 2:

        Theta_Leech(q) = E4(q)^3 - 720 Delta(q)

    and n(m) is the coefficient of q^m. The zero vector is q^0.
    The Leech lattice has no vectors in shell m=1.
    """
    if M < 0:
        raise ValueError("M must be non-negative")

    e4 = [0] * (M + 1)
    e4[0] = 1
    for n in range(1, M + 1):
        e4[n] = 240 * sigma_power(n, 3)

    e4_cubed = poly_pow_trunc(e4, 3, M)
    delta = delta_coefficients(M)
    theta = [e4_cubed[i] - 720 * delta[i] for i in range(M + 1)]
    return {m: theta[m] for m in range(2, M + 1)}


def cumulative_shell_counts(M: int) -> dict[int, int]:
    counts = leech_shell_counts(M)
    total = 0
    out = {}
    for m in range(2, M + 1):
        total += counts[m]
        out[m] = total
    return out


@dataclass(frozen=True)
class TrueLeechRate:
    M: int
    shell_counts: dict[int, int]
    cumulative_count: int
    shape_bits_per_vector: int
    shape_bits_per_dim: float
    gain_bits_per_vector: int
    total_bits_per_vector: int
    total_bits_per_dim: float


def true_leech_rate(M: int, gain_bits: int = 0) -> TrueLeechRate:
    counts = leech_shell_counts(M)
    cumulative = sum(counts.values())
    shape_bits = ceil(log2(cumulative)) if cumulative > 1 else 1
    total_bits = shape_bits + gain_bits
    return TrueLeechRate(
        M=M,
        shell_counts=counts,
        cumulative_count=cumulative,
        shape_bits_per_vector=shape_bits,
        shape_bits_per_dim=shape_bits / 24,
        gain_bits_per_vector=gain_bits,
        total_bits_per_vector=total_bits,
        total_bits_per_dim=total_bits / 24,
    )


class TrueLeechIndex:
    """
    Placeholder for paper-faithful index/deindex.

    This class intentionally does not pretend to quantize yet. The paper-faithful
    implementation still needs shell class leaders, class cardinalities, and
    rank/unrank for local Golay/permutation/sign symmetries.
    """

    def __init__(self, M: int):
        self.M = M
        self.shell_counts = leech_shell_counts(M)
        self.cumulative_counts = cumulative_shell_counts(M)
        self.total_count = self.cumulative_counts[M] if M >= 2 else 0

    def shell_for_index(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self.total_count:
            raise IndexError(index)
        previous = 0
        for shell, cumulative in self.cumulative_counts.items():
            if index < cumulative:
                return shell, index - previous
            previous = cumulative
        raise IndexError(index)

    def dequantize(self, index: int):
        raise NotImplementedError(
            "True Leech dequantization needs class leader tables and local "
            "symmetry unranking. This file currently implements exact shell "
            "counts/rates only."
        )

    def quantize(self, x):
        raise NotImplementedError(
            "True Adoul-Barth search needs class leader tables and local "
            "symmetry ranking. This file currently implements exact shell "
            "counts/rates only."
        )


def demo(M: int = 12, gain_bits: int = 1) -> None:
    rate = true_leech_rate(M, gain_bits=gain_bits)
    cumulative = cumulative_shell_counts(M)

    print(f"True Leech Lambda_24({M})")
    print("shell counts:")
    for shell, count in rate.shell_counts.items():
        print(f"  m={shell:2d}: n(m)={count}, N(m)={cumulative[shell]}")
    print(f"cumulative count N({M}): {rate.cumulative_count}")
    print(f"shape bits/vector: {rate.shape_bits_per_vector}")
    print(f"shape bits/dim: {rate.shape_bits_per_dim:.6f}")
    print(f"gain bits/vector: {rate.gain_bits_per_vector}")
    print(f"total bits/vector: {rate.total_bits_per_vector}")
    print(f"total bits/dim: {rate.total_bits_per_dim:.6f}")


def _argparse_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="True Leech shell counts and paper-compatible rates")
    parser.add_argument("--M", type=int, default=12)
    parser.add_argument("--gain_bits", type=int, default=1)
    args = parser.parse_args()
    demo(M=args.M, gain_bits=args.gain_bits)


if __name__ == "__main__":
    try:
        import fire
    except ImportError:
        _argparse_main()
    else:
        fire.Fire(demo)
