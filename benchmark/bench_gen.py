import argparse
from itertools import islice
import random
import os

import bench

def main():
    seed = os.environ.get("TAPP_BENCH_SEED")
    if seed is not None:
        random.seed(int(seed))

    parser = argparse.ArgumentParser(description="Generate benchmark contraction specifications")
    parser.add_argument("-pi", "--product", type=int, default=0, help="number of contracted indicies ((A xor B) and D)")
    parser.add_argument("-fai", "--free_a", type=int, default=0, help="number of free indicies in tensor A (A and D)")
    parser.add_argument("-fbi", "--free_b", type=int, default=0, help="number of free indicies in tensor B (B and D)")
    parser.add_argument("-hi", "--hadamard", type=int, default=0, help="number of hadamard indicies (A, B and D)")
    parser.add_argument("-rai", "--reduced_a", type=int, default=0, help="number of reduced indicies in tensor A (A)")
    parser.add_argument("-rbi", "--reduced_b", type=int, default=0, help="number of reduced indicies in tensor B (B)")
    parser.add_argument("-bi", "--broadcast", type=int, default=0, help="number of broadcast indicies (D)")
    parser.add_argument("-r", "--repeats", type=int, default=10, help="number repeated runs of the benchmark")

    args = parser.parse_args()
    nr_products, nr_free_a, nr_free_b, nr_hadamard, nr_reduced_a, nr_reduced_b, nr_broadcast, repeats = (
        args.product, args.free_a, args.free_b, args.hadamard, args.reduced_a, args.reduced_b, args.broadcast, args.repeats
    )

    print(f"Generating benchmark spec with {nr_products} contracted, {nr_free_a} free in A, {nr_free_b} free in B, "
            f"{nr_hadamard} hadamard, {nr_reduced_a} reduced in A, {nr_reduced_b} reduced in B, {nr_broadcast} broadcast indicies")

    # Generate indicies
    idx = [chr(ord('a') + i) for i in range(nr_products + nr_free_a + nr_free_b + nr_hadamard + nr_reduced_a + nr_reduced_b + nr_broadcast)]
    it = iter(idx)
    idx_product   = list(islice(it, nr_products))
    idx_free_a    = list(islice(it, nr_free_a))
    idx_free_b    = list(islice(it, nr_free_b))
    idx_hadamard  = list(islice(it, nr_hadamard))
    idx_reduced_a = list(islice(it, nr_reduced_a))
    idx_reduced_b = list(islice(it, nr_reduced_b))
    idx_broadcast = list(islice(it, nr_broadcast))

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
        "repeats": repeats,
    }
    bench.bench_by_specs([spec])

if __name__ == "__main__":
    main()