import argparse
import random
from itertools import islice

import bench


def _int_at_least(value, minimum):
    n = int(value)
    if n < minimum:
        raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {n}")
    return n


# Named functions rather than lambdas, since argparse shows the name in its
# "invalid <name> value" message for non-integer input.
def positive_int(value):
    return _int_at_least(value, 1)


def count(value):
    return _int_at_least(value, 0)


def main():
    bench.seed_from_env()

    parser = argparse.ArgumentParser(description="Generate benchmark contraction specifications")
    parser.add_argument("--product", type=count, default=0, help="number of contracted indices (A and B)")
    parser.add_argument("--free-a", type=count, default=0, help="number of free indices in tensor A (A and D)")
    parser.add_argument("--free-b", type=count, default=0, help="number of free indices in tensor B (B and D)")
    parser.add_argument("--hadamard", type=count, default=0, help="number of hadamard indices (A, B and D)")
    parser.add_argument("--reduced-a", type=count, default=0, help="number of reduced indices in tensor A (A)")
    parser.add_argument("--reduced-b", type=count, default=0, help="number of reduced indices in tensor B (B)")
    parser.add_argument("--broadcast", type=count, default=0, help="number of broadcast indices (D)")
    parser.add_argument("--repeats", type=positive_int, default=10, help="number of timed runs of the benchmark")

    args = parser.parse_args()

    print(
        f"Generating benchmark spec with {args.product} contracted, {args.free_a} free in A, {args.free_b} free in B, "
        f"{args.hadamard} hadamard, {args.reduced_a} reduced in A, {args.reduced_b} reduced in B, "
        f"{args.broadcast} broadcast indices"
    )

    # Generate indices
    counts = [args.product, args.free_a, args.free_b, args.hadamard, args.reduced_a, args.reduced_b, args.broadcast]
    idx = [chr(ord("a") + i) for i in range(sum(counts))]
    it = iter(idx)
    idx_product = list(islice(it, args.product))
    idx_free_a = list(islice(it, args.free_a))
    idx_free_b = list(islice(it, args.free_b))
    idx_hadamard = list(islice(it, args.hadamard))
    idx_reduced_a = list(islice(it, args.reduced_a))
    idx_reduced_b = list(islice(it, args.reduced_b))
    idx_broadcast = list(islice(it, args.broadcast))

    # Same notation as the indices and extents fields in contractions.txt
    idx_a = "".join(idx_product + idx_free_a + idx_hadamard + idx_reduced_a)
    idx_b = "".join(idx_product + idx_free_b + idx_hadamard + idx_reduced_b)
    idx_d = "".join(idx_free_a + idx_free_b + idx_hadamard + idx_broadcast)
    indices = f"{idx_a}-{idx_b}-{idx_d}"
    extents = {i: random.randint(10, 20) for i in idx}
    extents_str = " ".join(f"{i}:{e}" for i, e in extents.items())

    spec = {
        "name": f"{indices} {extents_str}",
        "indices": indices,
        "extents": extents,
        **{f"datatype_{t}": "f32" for t in ("a", "b", "c", "d")},
        **{f"op_{t}": "identity" for t in ("a", "b", "c", "d")},
        "precision": "default",
        "alpha": random.uniform(-10, 10),
        "beta": random.uniform(-10, 10),
        "repeats": args.repeats,
    }
    bench.bench_by_specs([spec])


if __name__ == "__main__":
    main()
