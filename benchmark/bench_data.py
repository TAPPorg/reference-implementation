# FLOP count, data movement, and arithmetic intensity for the TAPP operation
#
#     D <- alpha * A * B + beta * C
#
# A and B are input tensors, C is an input tensor with the same shape as D,
# and D is the output tensor.
#
# Index classification
# --------------------
# Every index belongs to exactly one of the following sets, determined by
# which tensors it appears in (C is treated as D):
#
#     FA : free in A        - appears in A and D, not in B
#     FB : free in B        - appears in B and D, not in A
#     IA : isolated in A    - appears only in A (summed out)
#     IB : isolated in B    - appears only in B (summed out)
#     H  : Hadamard         - appears in A, B, and D
#     P  : contracted       - appears in A and B, not in D
#     X  : broadcast        - appears only in D
#
# For each set, the corresponding lowercase symbol is the product of the
# extents of its indices; an empty set gives 1:
#
#     f_a = prod(FA),  f_b = prod(FB),  i_a = prod(IA),  i_b = prod(IB),
#     h   = prod(H),   p   = prod(P),   x   = prod(X)
#
# Repeated indices within a single tensor (diagonals/traces) are not covered.
#
# FLOP count
# ----------
# Counts the minimal number of real arithmetic operations, assuming general
# alpha and beta (no special-casing of 0 or 1). The isolated indices are
# reduced once, before the contraction, rather than inside its inner loop.
#
#     flops = h*p*((i_a - 1)*f_a + (i_b - 1)*f_b)    # reduce IA in A, IB in B
#           + 2*p*f_a*f_b*h                          # contraction + alpha scaling
#           + 2*x*f_a*f_b*h                          # beta*C + addition, per element of D
#
# Summing n elements costs n - 1 additions, so with no isolated, Hadamard, or
# broadcast indices this reduces to the standard GEMM count 2mnk + 2mn.
#
# For complex arithmetic, each operation is weighted by its cost in real
# operations: a complex addition is 2, and a multiply-add is 6 + 2 = 8, i.e.
# 4x its real count of 2. The reduction term is scaled by 2 and the other two
# terms by 4, matching the usual 8mnk convention for complex GEMM.
# Conjugation is free. Mixed real/complex operands are counted as fully
# complex, which slightly over-counts them (real * complex is 2, not 6).
#
# Data movement
# -------------
# Compulsory traffic, in elements: each tensor is read or written exactly once.
# Multiply by the element size to get bytes. This is a lower bound; cache
# misses and write-allocate can make actual traffic higher.
#
#     elements = f_a*i_a*h*p          # read A
#              + f_b*i_b*h*p          # read B
#              + 2*f_a*f_b*h*x        # read C, write D
#
# Arithmetic intensity
# --------------------
# FLOPs per element moved. The Hadamard extent h cancels out, since Hadamard
# indices add no data reuse:
#
#     intensity = (2*f_a*f_b*(p + x) + p*((i_a - 1)*f_a + (i_b - 1)*f_b))
#               / (p*(f_a*i_a + f_b*i_b) + 2*f_a*f_b*x)
#
# Divide by the element size to get FLOPs per byte.

import math
from typing import NamedTuple

import bench_helpers


class ExtentProducts(NamedTuple):
    # Product of the extents of each index class; see the header comment.
    f_a: int
    f_b: int
    i_a: int
    i_b: int
    h: int
    p: int
    x: int


def classify_indices(idx_a, idx_b, idx_d):
    d_set = set(idx_d)
    f_a = [c for c in idx_a if c not in idx_b and c in d_set]
    f_b = [c for c in idx_b if c not in idx_a and c in d_set]
    i_a = [c for c in idx_a if c not in idx_b and c not in d_set]
    i_b = [c for c in idx_b if c not in idx_a and c not in d_set]
    h = [c for c in idx_a if c in idx_b and c in d_set]
    p = [c for c in idx_a if c in idx_b and c not in d_set]
    x = [c for c in idx_d if c not in idx_a and c not in idx_b]
    return f_a, f_b, i_a, i_b, h, p, x


def extent_products(idx_a, idx_b, idx_d, ext_a, ext_b, ext_d):
    idx_fa, idx_fb, idx_ia, idx_ib, idx_h, idx_p, idx_x = classify_indices(idx_a, idx_b, idx_d)
    lookup_a = dict(zip(idx_a, ext_a))
    lookup_b = dict(zip(idx_b, ext_b))
    lookup_d = dict(zip(idx_d, ext_d))
    return ExtentProducts(
        f_a=math.prod(bench_helpers.lookup(lookup_a, c, "extent") for c in idx_fa),
        f_b=math.prod(bench_helpers.lookup(lookup_b, c, "extent") for c in idx_fb),
        i_a=math.prod(bench_helpers.lookup(lookup_a, c, "extent") for c in idx_ia),
        i_b=math.prod(bench_helpers.lookup(lookup_b, c, "extent") for c in idx_ib),
        h=math.prod(bench_helpers.lookup(lookup_a, c, "extent") for c in idx_h),
        p=math.prod(bench_helpers.lookup(lookup_a, c, "extent") for c in idx_p),
        x=math.prod(bench_helpers.lookup(lookup_d, c, "extent") for c in idx_x),
    )


def FLOPs_raw(e, is_complex=False):
    add, mul_add = (2, 4) if is_complex else (1, 1)
    return (
        add * e.h * e.p * ((e.i_a - 1) * e.f_a + (e.i_b - 1) * e.f_b)
        + mul_add * 2 * e.p * e.f_a * e.f_b * e.h
        + mul_add * 2 * e.x * e.f_a * e.f_b * e.h
    )


def data_moved_raw(e):
    return e.f_a * e.i_a * e.h * e.p + e.f_b * e.i_b * e.h * e.p + 2 * e.f_a * e.f_b * e.h * e.x


def intensity_raw(e, is_complex=False):
    return FLOPs_raw(e, is_complex) / data_moved_raw(e)


def FLOPs(idx_a, idx_b, idx_d, ext_a, ext_b, ext_d, is_complex=False):
    return FLOPs_raw(extent_products(idx_a, idx_b, idx_d, ext_a, ext_b, ext_d), is_complex)


def data_moved(idx_a, idx_b, idx_d, ext_a, ext_b, ext_d):
    return data_moved_raw(extent_products(idx_a, idx_b, idx_d, ext_a, ext_b, ext_d))


def intensity(idx_a, idx_b, idx_d, ext_a, ext_b, ext_d, is_complex=False):
    return intensity_raw(extent_products(idx_a, idx_b, idx_d, ext_a, ext_b, ext_d), is_complex)
