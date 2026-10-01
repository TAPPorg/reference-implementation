"""Builds benchmarks from contraction specs and runs them against a
TAPP-conformant shared library, via tapp_bindings.py.

CPU only: build_data() and build_scalar() allocate every operand (A/B/C/D)
and scalar (alpha/beta) as a plain ctypes array, which lives in this Python
process's own host RAM. Those host addresses are passed straight through to
TAPP_execute_product as void* (see tapp_bindings.py), with no device
allocation or host<->device transfer anywhere in the benchmark. A TAPP
implementation that expects device (e.g. GPU) memory for its operands will
not work with this benchmark.
"""

import random
import time
from dataclasses import dataclass

import numpy as np

import tapp_bindings
import bench_data
import bench_helpers

@dataclass
class Benchmark:
    # Per-tensor fields are dicts keyed by "a", "b", "c", "d".
    name: str
    indices: dict
    extents: dict
    strides: dict
    datatypes: dict
    ops: dict
    data: dict
    alpha: object  # ctypes array from build_scalar
    beta: object
    prec: int
    repeats: int = 1

def bench_by_specs(specs):
    # Builds, runs and prints one spec at a time, so only one benchmark's
    # buffers are allocated at once and a bad spec is reported as SKIPPED.
    handle = tapp_bindings.create_handle()
    executor = tapp_bindings.create_executor()
    results = []
    try:
        for spec in specs:
            try:
                benchmark = build_benchmark(spec)
            except ValueError as error:
                result = (None, None, None, error)
            else:
                result = run_contraction(benchmark, executor, handle)
            print_result(spec["name"], result)
            results.append(result)
    finally:
        tapp_bindings.destroy_executor(executor)
        tapp_bindings.destroy_handle(handle)
    return results

def print_result(name, result):
    min_time, avg_time, gflops, error = result
    if error is None:
        print(f"{name}: {min_time:.6f} s (min), {avg_time:.6f} s (avg), {gflops:.3f} GFLOP/s")
    else:
        print(f"{name}: SKIPPED ({error})")

def tensor_size(extents, strides):
    return 1 + sum(abs(s * (e - 1)) for s, e in zip(strides, extents))

def calc_strides(ext):
    strides, stride = [], 1
    for e in ext:
        strides.append(stride)
        stride *= e
    return strides

def build_data(size, datatype):
    # Allocates in host RAM -- see the "CPU only" note in the module docstring.
    # numpy is seeded from Python's random, so TAPP_BENCH_SEED still makes runs repeatable.
    ctype, components = bench_helpers.lookup(tapp_bindings.CTYPES_BY_NAME, datatype.lower(), "datatype")
    n = size * components
    rng = np.random.default_rng(random.getrandbits(64))
    values = rng.uniform(-10, 10, n).astype(np.dtype(ctype))
    return (ctype * n).from_buffer(values)  # shares memory with values and keeps it alive

def build_scalar(value, datatype):
    # Allocates in host RAM -- see the "CPU only" note in the module docstring.
    #
    # Returned as a 1- or 2-element(for complex) array rather than a bare
    # ctypes scalars because ctypes only auto-converts arrays to void* -- a bare
    # scalar would need an explicit byref().
    ctype, components = bench_helpers.lookup(tapp_bindings.CTYPES_BY_NAME, datatype.lower(), "datatype")
    if components == 2:
        values = (value.real, value.imag)
    else:
        if value.imag != 0:
            raise ValueError(f"non-zero imaginary part not representable in real datatype {datatype!r}")
        values = (value.real,)
    return (ctype * components)(*values)

def build_benchmark(spec):
    # spec is a dict in the format returned by bench_file.parse_contractions.
    idx_a, idx_b, idx_d = spec["indices"].split("-")
    indices = {"a": idx_a, "b": idx_b, "c": idx_d, "d": idx_d}
    extents, strides, datatypes, ops, data = {}, {}, {}, {}, {}
    for t in indices:
        extents[t] = [bench_helpers.lookup(spec["extents"], i, "extent for index") for i in indices[t]]
        strides[t] = calc_strides(extents[t])
        datatypes[t] = bench_helpers.lookup(tapp_bindings.DATATYPE_BY_NAME, spec[f"datatype_{t}"].lower(), "datatype")
        ops[t] = bench_helpers.lookup(tapp_bindings.OP_BY_NAME, spec[f"op_{t}"].lower(), "op")
        data[t] = build_data(tensor_size(extents[t], strides[t]), spec[f"datatype_{t}"])
    alpha = build_scalar(spec["alpha"], spec["datatype_d"])
    beta = build_scalar(spec["beta"], spec["datatype_d"])
    prec = bench_helpers.lookup(tapp_bindings.PRECTYPE_BY_NAME, spec["precision"].lower(), "precision")
    return Benchmark(spec["name"], indices, extents, strides, datatypes, ops, data, alpha, beta, prec, spec["repeats"])

def time_execution(plan, executor, alpha, A, B, beta, C, D, repeats):
    tapp_bindings.execute_product(plan, executor, alpha, A, B, beta, C, D)  # warm-up, untimed

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        tapp_bindings.execute_product(plan, executor, alpha, A, B, beta, C, D)
        times.append(time.perf_counter() - start)
    return min(times), sum(times) / len(times)

def run_contraction(benchmark, executor, handle):
    b = benchmark
    plan = None
    infos = {}
    try:
        infos = {t:tapp_bindings.create_tensor_info(b.datatypes[t], b.extents[t], b.strides[t]) for t in ("a", "b", "c", "d")}

        plan = tapp_bindings.create_tensor_product(
            handle,
            b.ops["a"], infos["a"], b.indices["a"],
            b.ops["b"], infos["b"], b.indices["b"],
            b.ops["c"], infos["c"], b.indices["c"],
            b.ops["d"], infos["d"], b.indices["d"],
            b.prec,
        )

        time_min, time_avg = time_execution(plan, executor, b.alpha, b.data["a"], b.data["b"], b.beta, b.data["c"], b.data["d"], b.repeats)

        is_complex = any(dt in (tapp_bindings.TAPPDataType.C32, tapp_bindings.TAPPDataType.C64) for dt in b.datatypes.values())
        flops = bench_data.FLOPs(b.indices["a"], b.indices["b"], b.indices["d"], b.extents["a"], b.extents["b"], b.extents["d"], is_complex)
        gflops = flops / time_min / 1e9

        result = (time_min, time_avg, gflops, None)
    except (tapp_bindings.TAPPError, ValueError) as error:
        result = (None, None, None, error)
    finally:
        if plan is not None:
            tapp_bindings.destroy_tensor_product(plan)
        for info in infos.values():
            tapp_bindings.destroy_tensor_info(info)
    return result
