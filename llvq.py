import numpy as np
import math
from collections import defaultdict
from itertools import product
from dataclasses import dataclass

# ============================================================
# GOLAY CODE
# ============================================================

class GolayCode:
    """
    Extended binary Golay code (24,12,8)
    """

    def __init__(self):

        self.G = self._generator_matrix()

        self.codewords = self._generate_codewords()

    def _generator_matrix(self):

        I = np.eye(12, dtype=np.uint8)

        P = np.array([
            [1,1,1,1,1,0,0,0,1,0,0,0],
            [1,1,1,0,0,1,0,1,0,1,0,0],
            [1,1,1,0,0,0,1,0,0,0,1,0],
            [1,1,0,1,0,1,1,0,0,0,0,1],
            [1,0,1,1,0,1,0,0,0,1,1,0],
            [1,0,0,0,1,1,1,0,1,0,1,0],
            [0,1,1,1,1,1,0,1,0,0,0,0],
            [0,1,1,0,1,0,1,0,1,1,0,0],
            [0,1,0,1,0,1,1,1,0,1,0,0],
            [0,0,1,0,1,1,0,1,1,0,1,0],
            [0,0,0,1,1,0,1,1,1,0,0,1],
            [0,0,0,0,0,0,0,0,1,1,1,1],
        ], dtype=np.uint8)

        return np.concatenate([I, P], axis=1)

    def encode(self, msg):

        msg = np.asarray(msg, dtype=np.uint8)

        return (msg @ self.G) % 2

    def _generate_codewords(self):

        out = []

        for i in range(1 << 12):

            bits = np.array(
                [(i >> j) & 1 for j in range(12)],
                dtype=np.uint8
            )

            out.append(self.encode(bits))

        return np.array(out, dtype=np.uint8)


GOLAY = GolayCode()

# ============================================================
# LEECH CONSTRUCTION A
# ============================================================

def is_valid_leech_vector(x):
    """
    Simplified Conway-Sloane parity checks.
    """

    x = np.asarray(x)

    # all coordinates same parity
    parity = x[0] & 1

    if not np.all((x & 1) == parity):
        return False

    # coordinate sum divisible by 4
    if np.sum(x) % 4 != 0:
        return False

    return True


def construction_A_vectors(max_coord=4):
    """
    Generate Leech-style vectors using Golay code.
    """

    vals = np.arange(-max_coord, max_coord + 1)

    for cw in GOLAY.codewords:

        parity = cw.astype(np.int32)

        for base in product(vals, repeat=24):

            x = np.array(base, dtype=np.int32)

            if np.all((x & 1) == parity):

                if is_valid_leech_vector(x):

                    yield x


# ============================================================
# SHELLS
# ============================================================

@dataclass
class Shell:
    m: int
    vectors: list


class ShellDatabase:

    def __init__(self):

        self.shells = {}
        self.offsets = {}

        self.all_vectors = []
        self.total_count = 0

    def build(self, M, max_coord=4):

        shell_map = defaultdict(list)

        print("Generating vectors...")

        for v in construction_A_vectors(max_coord):

            n2 = np.sum(v * v)

            if n2 < 4:
                continue

            if n2 > 2 * M:
                continue

            m = n2 // 2

            shell_map[m].append(v)

        offset = 0

        for m in sorted(shell_map.keys()):

            vecs = shell_map[m]

            self.shells[m] = Shell(m, vecs)

            self.offsets[m] = offset

            offset += len(vecs)

            self.all_vectors.extend(vecs)

        self.total_count = offset

    def shell_sizes(self):

        return {
            m: len(shell.vectors)
            for m, shell in self.shells.items()
        }


# ============================================================
# QUANTIZER
# ============================================================

class LLVQ:

    def __init__(self, shell_db):

        self.db = shell_db

    # ========================================================
    # INDEXING
    # ========================================================

    def index_to_vector(self, idx):

        return self.db.all_vectors[idx]

    def vector_to_index(self, vec):

        vec = tuple(vec.tolist())

        for i, v in enumerate(self.db.all_vectors):

            if tuple(v.tolist()) == vec:
                return i

        raise ValueError("vector not found")

    # ========================================================
    # DISTANCES
    # ========================================================

    def euclidean_distance(self, x, v):

        return np.sum((x - v) ** 2)

    def angular_distance(self, x, v):

        xn = np.linalg.norm(x)
        vn = np.linalg.norm(v)

        if xn == 0 or vn == 0:
            return 1.0

        return 1.0 - np.dot(x, v) / (xn * vn)

    # ========================================================
    # QUANTIZATION
    # ========================================================

    def quantize(self, x, mode="euclidean"):

        x = np.asarray(x, dtype=np.float64)

        best_idx = None
        best_v = None
        best_score = 1e30

        for idx, v in enumerate(self.db.all_vectors):

            if mode == "euclidean":

                d = self.euclidean_distance(x, v)

            elif mode == "angular":

                d = self.angular_distance(x, v)

            else:
                raise ValueError(mode)

            if d < best_score:

                best_score = d
                best_idx = idx
                best_v = v

        return best_idx, best_v, best_score

    # ========================================================
    # DEQUANTIZATION
    # ========================================================

    def dequantize(self, idx):

        return self.index_to_vector(idx)


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Build finite spherical codebook
    # --------------------------------------------------------

    M = 20

    db = ShellDatabase()

    db.build(M=M, max_coord=2)

    print("Shell sizes:")
    print(db.shell_sizes())

    print("Total vectors:", db.total_count)

    # --------------------------------------------------------
    # Quantizer
    # --------------------------------------------------------

    q = LLVQ(db)

    x = np.random.randn(24) * 3

    idx, v, score = q.quantize(
        x,
        mode="euclidean"
    )

    print("\nQUANTIZATION")
    print("index:", idx)
    print("score:", score)

    print("\nnearest vector:")
    print(v)

    # --------------------------------------------------------
    # Dequantization
    # --------------------------------------------------------

    recon = q.dequantize(idx)

    print("\nDEQUANTIZATION")
    print(recon)

    print("\nExact reconstruction:",
          np.all(recon == v))
