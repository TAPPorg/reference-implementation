"""Runs the tensor contractions described in a contractions.txt-format file
(see that file for the config format) against a TAPP-conformant shared
library, via tapp_bindings.py, and reports timing/GFLOP-s per contraction.

Operands are allocated in host memory only; see bench.py.
"""

import argparse
import configparser
from pathlib import Path

import bench

DEFAULT_CONTRACTIONS_FILE = Path(__file__).parent / "contractions.txt"


def parse_contractions(path):
    cp = configparser.ConfigParser()
    with open(path) as f:
        cp.read_file(f)
    specs = []
    for name in cp.sections():
        sec = cp[name]
        indices = sec["indices"]
        extents = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in sec["extents"].split()}
        dtype = sec.get("datatype", "f32")
        spec = {
            "name": name,
            "indices": indices,
            "extents": extents,
            "datatype_a": sec.get("datatype_a", dtype),
            "datatype_b": sec.get("datatype_b", dtype),
            "datatype_c": sec.get("datatype_c", dtype),
            "datatype_d": sec.get("datatype_d", dtype),
            "op_a": sec.get("op_a", "identity"),
            "op_b": sec.get("op_b", "identity"),
            "op_c": sec.get("op_c", "identity"),
            "op_d": sec.get("op_d", "identity"),
            "precision": sec.get("precision", "default"),
            "alpha": complex(sec.get("alpha", "1.0")),
            "beta": complex(sec.get("beta", "0.0")),
            "repeats": int(sec.get("repeats", 10)),
        }
        specs.append(spec)
    return specs


def main():
    bench.seed_from_env()

    parser = argparse.ArgumentParser(description="Run the contractions in a contractions.txt-format file")
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=DEFAULT_CONTRACTIONS_FILE,
        help="contractions file (default: contractions.txt next to this script)",
    )
    args = parser.parse_args()

    try:
        specs = parse_contractions(args.file)
    except FileNotFoundError:
        parser.error(f"contractions file not found: {args.file}")
    if not specs:
        parser.error(f"no contractions found in {args.file}")
    bench.bench_by_specs(specs)


if __name__ == "__main__":
    main()
