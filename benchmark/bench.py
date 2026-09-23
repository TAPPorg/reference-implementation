"""Runs the tensor contractions described in a contractions.txt-format file
(see that file for the config format) against a TAPP-conformant shared
library, via tapp_bindings.py, and reports timing/GFLOP-s per contraction.

CPU only: build_data() and build_scalar() below allocate every operand
(A/B/C/D) and scalar (alpha/beta) as a plain ctypes array, which lives in
this Python process's own host RAM. Those host addresses are passed
straight through to TAPP_execute_product as void* (see tapp_bindings.py),
with no device allocation or host<->device transfer anywhere in this file.
A TAPP implementation that expects device (e.g. GPU) memory for its operands
will not work with this benchmark.
"""

import configparser
import ctypes
import os
import random
import sys
import time
from pathlib import Path

import tapp_bindings

DEFAULT_CONTRACTIONS_FILE = Path(__file__).parent / "contractions.txt"

def lookup(table, key, what):
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"unknown {what} {key!r}") from None

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
            "repeats": int(sec.get("repeats", 1)),
        }
        specs.append(spec)
    return specs

def tensor_size(extents, strides):
    return 1 + sum(abs(s * (e - 1)) for s, e in zip(strides, extents))

def build_tensor_info(idx, extents, datatype):
    ext = [lookup(extents, letter, "extent for index") for letter in idx]
    strides, stride = [], 1
    for e in ext:
        strides.append(stride)
        stride *= e
    info = tapp_bindings.create_tensor_info(
        datatype=lookup(tapp_bindings.DATATYPE_BY_NAME, datatype.lower(), "datatype"),
        extents=(ctypes.c_int64 * len(ext))(*ext),
        strides=(ctypes.c_int64 * len(strides))(*strides),
    )
    return info, tensor_size(ext, strides)

def build_data(size, datatype):
    # Allocates in host RAM -- see the "CPU only" note in the module docstring.
    ctype, components = lookup(tapp_bindings.CTYPES_BY_NAME, datatype.lower(), "datatype")
    n = size * components
    return (ctype * n)(*(random.uniform(-10, 10) for _ in range(n)))

def build_scalar(value, datatype):
    # Allocates in host RAM -- see the "CPU only" note in the module docstring.
    #
    # Returned as a 1- or 2-element(for complex) array rather than a bare
    # ctypes scalars because ctypes only auto-converts arrays to void* -- a bare
    # scalar would need an explicit byref().
    ctype, components = lookup(tapp_bindings.CTYPES_BY_NAME, datatype.lower(), "datatype")
    if components == 2:
        values = (value.real, value.imag)
    else:
        if value.imag != 0:
            raise ValueError(f"non-zero imaginary part not representable in real datatype {datatype!r}")
        values = (value.real,)
    return (ctype * components)(*values)

def is_complex_datatype(datatype):
    return lookup(tapp_bindings.CTYPES_BY_NAME, datatype.lower(), "datatype")[1] == 2

def mul_flops(is_complex_1, is_complex_2):
    # real * real = 1 flop. real * complex = 2 muls, no cross terms = 2 flops.
    # complex * complex, (a+bi)(c+di) = (ac-bd) + (ad+bc)i = 4 muls + 2 add/sub = 6 flops.
    return (1, 2, 6)[is_complex_1 + is_complex_2]

def add_flops(is_complex):
    # real + real = 1 flop. Anything landing in a complex accumulator = 2 flops
    # (real part and imaginary part added separately).
    return 2 if is_complex else 1

def estimate_flops(spec):
    idx_a, idx_b, idx_d = spec["indices"].split("-")
    extents = spec["extents"]
    d_set = set(idx_d)

    def group_size(letters):
        size = 1
        for letter in letters:
            size *= lookup(extents, letter, "extent for index")
        return size

    def unary_reduction_flops(idx_this, idx_other, is_complex):
        unary = [c for c in idx_this if c not in idx_other and c not in d_set]
        free = [c for c in idx_this if c in d_set]
        return (group_size(unary) - 1) * group_size(free) * add_flops(is_complex)

    contracted = [c for c in idx_a if c in idx_b and c not in d_set]
    result_size = group_size(idx_d)
    binary_size = group_size(contracted)

    is_complex_a = is_complex_datatype(spec["datatype_a"])
    is_complex_b = is_complex_datatype(spec["datatype_b"])
    is_complex_c = is_complex_datatype(spec["datatype_c"])
    is_complex_d = is_complex_datatype(spec["datatype_d"])
    is_complex_alpha = is_complex_d
    is_complex_beta = is_complex_d
    # A * B is complex if either operand is; its running sum is then complex too.
    is_complex_ab = is_complex_a or is_complex_b

    flops = result_size * (mul_flops(is_complex_beta, is_complex_c)
                            + add_flops(is_complex_d))                    # beta * C, combined into D
    flops += unary_reduction_flops(idx_a, idx_b, is_complex_a)            # unary contraction over A
    flops += unary_reduction_flops(idx_b, idx_a, is_complex_b)            # unary contraction over B
    flops += result_size * binary_size * (mul_flops(is_complex_a, is_complex_b)
                                           + add_flops(is_complex_ab))     # A * B
    flops += result_size * mul_flops(is_complex_alpha, is_complex_ab)     # alpha * (...)
    return flops

def time_execution(plan, executor, alpha, A, B, beta, C, D, repeats):
    tapp_bindings.execute_product(plan, executor, alpha, A, B, beta, C, D)  # warm-up, untimed

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        tapp_bindings.execute_product(plan, executor, alpha, A, B, beta, C, D)
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times)

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONTRACTIONS_FILE
    specs = parse_contractions(path)

    seed = os.environ.get("TAPP_BENCH_SEED")
    if seed is not None:
        random.seed(int(seed))

    handle = tapp_bindings.create_handle()
    executor = tapp_bindings.create_executor()
    
    for spec in specs:
        idx_a, idx_b, idx_d = spec["indices"].split("-")
        idx = {"a": idx_a, "b": idx_b, "c": idx_d, "d": idx_d}

        infos = {}
        plan = None
        try:
            data = {}
            for t in ("a", "b", "c", "d"):
                infos[t], size = build_tensor_info(idx[t], spec["extents"], spec[f"datatype_{t}"])
                data[t] = build_data(size, spec[f"datatype_{t}"])

            plan = tapp_bindings.create_tensor_product(
                handle,
                lookup(tapp_bindings.OP_BY_NAME, spec["op_a"].lower(), "op"), infos["a"], idx["a"],
                lookup(tapp_bindings.OP_BY_NAME, spec["op_b"].lower(), "op"), infos["b"], idx["b"],
                lookup(tapp_bindings.OP_BY_NAME, spec["op_c"].lower(), "op"), infos["c"], idx["c"],
                lookup(tapp_bindings.OP_BY_NAME, spec["op_d"].lower(), "op"), infos["d"], idx["d"],
                lookup(tapp_bindings.PRECTYPE_BY_NAME, spec["precision"].lower(), "precision"),
            )

            alpha = build_scalar(spec["alpha"], spec["datatype_d"])
            beta = build_scalar(spec["beta"], spec["datatype_d"])

            execution_time = time_execution(
                plan, executor,
                alpha, data["a"], data["b"],
                beta, data["c"], data["d"],
                spec["repeats"],
            )

            flops = estimate_flops(spec)
            gflops = flops / execution_time[0] / 1e9

            print(f"{spec['name']}: {execution_time[0]:.6f} s (min), {execution_time[1]:.6f} s (avg), {gflops:.3f} GFLOP/s")
        except (tapp_bindings.TAPPError, ValueError) as error:
            print(f"{spec['name']}: SKIPPED ({error})")
        finally:
            if plan is not None:
                tapp_bindings.destroy_tensor_product(plan)
            for info in infos.values():
                tapp_bindings.destroy_tensor_info(info)

    tapp_bindings.destroy_handle(handle)
    tapp_bindings.destroy_executor(executor)

if __name__ == "__main__":
    main()