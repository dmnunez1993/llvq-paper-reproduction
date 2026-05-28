from __future__ import annotations

from impl.leech_lattice_vector_quantizer import check_demo, demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Implicit Leech shell/class/local codeword database")
    parser.add_argument("--max-shell", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--indices", type=str, default="0,1103,1104,196559,196560")
    args = parser.parse_args()
    if args.check:
        indices = [int(part) for part in args.indices.split(",") if part.strip()]
        check_demo(max_shell=args.max_shell, indices=indices)
    else:
        demo(
            max_shell=args.max_shell,
            samples=args.samples,
            seed=args.seed,
            max_candidates=args.max_candidates,
        )


if __name__ == "__main__":
    main()
