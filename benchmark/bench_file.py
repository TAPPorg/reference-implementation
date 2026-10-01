"""Runs the tensor contractions described in a contractions.txt-format file
(see that file for the config format) against a TAPP-conformant shared
library, via tapp_bindings.py, and reports timing/GFLOP-s per contraction.

Operands are allocated in host memory only; see bench.py.
"""

import configparser
import os
import random
import sys
from pathlib import Path

import bench

DEFAULT_CONTRACTIONS_FILE = Path(__file__).parent / "contractions.txt"

def parse_contractions(path):
    cp = configparser.ConfigParser()
    cp.read(path)
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
    seed = os.environ.get("TAPP_BENCH_SEED")
    if seed is not None:
        random.seed(int(seed))
    
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONTRACTIONS_FILE
    specs = parse_contractions(path)
    
    bench.bench_by_specs(specs)

if __name__ == "__main__":
    main()