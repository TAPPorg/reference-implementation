"""ctypes bindings for TAPP-conformant shared libraries, covering the API
surface needed to build a tensor_info/tensor_product plan and execute a
single (non-batched) product.

TAPP_LIBRARY_PATH environment variable is used to set the shared library path.
If unset, the default is to look for libtapp-reference.{so,dylib} under a
build directory in the repo root.

Known limitations of this binding layer (some caused by the working state of
the TAPP standard):
  - Storage datatypes: only f32/f64/c32/c64 are usable (see
    DATATYPE_BY_NAME/CTYPES_BY_NAME below). TAPP_F16 and TAPP_BF16 are
    listed in TAPPDataType for completeness but have no ctypes mapping --
    ctypes has no native half-precision float type, so binding them would
    need a manual 16-bit encode/decode step that hasn't been written.
  - No batched product support: TAPP_execute_batched_product isn't bound,
    only the single-product TAPP_execute_product.
  - No tensor_info accessors: TAPP_get/set_nmodes, TAPP_get/set_extents,
    and TAPP_get/set_strides (tensor.h) aren't bound. Tensor shape is only
    ever set once, at TAPP_create_tensor_info time.
  - No status/attribute API: TAPP_destroy_status (status.h) and the whole
    attribute API, TAPP_attr_set/get/clear (attributes.h), aren't bound.
    execute_product() below deliberately leaves its per-call status handle
    unfreed rather than binding TAPP_destroy_status (see its comment for
    why); TAPP_attr_* has no caller anywhere in this repo yet.
  - CPU only: A/B/C/D and alpha/beta are always plain host-memory ctypes
    buffers passed straight through as void* -- there's no device
    allocation or host<->device transfer step anywhere in this file. A
    TAPP implementation backed by GPU/device memory (expecting device
    pointers) will not work through these bindings.
"""

import ctypes
import os
from pathlib import Path

TAPP_handle = ctypes.c_ssize_t
TAPP_executor = ctypes.c_ssize_t
TAPP_tensor_info = ctypes.c_ssize_t
TAPP_tensor_product = ctypes.c_ssize_t
TAPP_status = ctypes.c_ssize_t


class TAPPError(RuntimeError):
    """Raised when a TAPP_error return code indicates failure (TAPP_check_success == false)."""


class TAPPDataType:
    F32 = 0
    F64 = 1
    C32 = 2
    C64 = 3
    F16 = 4
    BF16 = 5


# F16/BF16 deliberately omitted -- see "Known limitations" in the module
# docstring above.
DATATYPE_BY_NAME = {
    "f32": TAPPDataType.F32,
    "f64": TAPPDataType.F64,
    "c32": TAPPDataType.C32,
    "c64": TAPPDataType.C64,
}

# (type(ctypes), components per tensor element(1 for real, 2 for complex)) for
# each storage datatype. c32/c64 are complex: per datatype.h, "stored with
# consecutive real and imaginary parts packed into 8/16 bytes" -- i.e. laid
# out as 2 consecutive real components, matching how a C99 
# `float complex`/`double complex` sits in memory.
CTYPES_BY_NAME = {
    "f32": (ctypes.c_float, 1),
    "f64": (ctypes.c_double, 1),
    "c32": (ctypes.c_float, 2),
    "c64": (ctypes.c_double, 2),
}


class TAPPPrecType:
    DEFAULT = -1
    F32F32_ACCUM_F32 = TAPPDataType.F32
    F64F64_ACCUM_F64 = TAPPDataType.F64
    F16F16_ACCUM_F16 = TAPPDataType.F16
    F16F16_ACCUM_F32 = 5
    BF16BF16_ACCUM_F32 = 6


PRECTYPE_BY_NAME = {
    "default": TAPPPrecType.DEFAULT,
    "f32f32_accum_f32": TAPPPrecType.F32F32_ACCUM_F32,
    "f64f64_accum_f64": TAPPPrecType.F64F64_ACCUM_F64,
    "f16f16_accum_f16": TAPPPrecType.F16F16_ACCUM_F16,
    "f16f16_accum_f32": TAPPPrecType.F16F16_ACCUM_F32,
    "bf16bf16_accum_f32": TAPPPrecType.BF16BF16_ACCUM_F32,
}


class TAPPElementOp:
    IDENTITY = 0
    CONJUGATE = 1


OP_BY_NAME = {
    "identity": TAPPElementOp.IDENTITY,
    "conjugate": TAPPElementOp.CONJUGATE,
}


def _find_library():
    override = os.environ.get("TAPP_LIBRARY_PATH")
    if override:
        return Path(override)

    repo_root = Path(__file__).resolve().parent.parent
    for build_dir in ("build", "cmake-build-debug", "cmake-build-release"):
        for name in ("libtapp-reference.so", "libtapp-reference.dylib"):
            candidate = repo_root / build_dir / name
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not find libtapp-reference.{so,dylib} under a build directory. "
        "Build it first (e.g. `cmake -S . -B build && cmake --build build`), "
        "or set the TAPP_LIBRARY_PATH environment variable to its full path."
    )


_lib = ctypes.CDLL(str(_find_library()))


def _proto(name, argtypes, restype):
    fn = getattr(_lib, name)
    fn.argtypes = argtypes
    fn.restype = restype
    return fn


_TAPP_create_handle = _proto("TAPP_create_handle", [ctypes.POINTER(TAPP_handle)], ctypes.c_int)
_TAPP_destroy_handle = _proto("TAPP_destroy_handle", [TAPP_handle], ctypes.c_int)

_TAPP_create_executor = _proto("TAPP_create_executor", [ctypes.POINTER(TAPP_executor)], ctypes.c_int)
_TAPP_destroy_executor = _proto("TAPP_destroy_executor", [TAPP_executor], ctypes.c_int)

_TAPP_create_tensor_info = _proto(
    "TAPP_create_tensor_info",
    [
        ctypes.POINTER(TAPP_tensor_info),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ],
    ctypes.c_int,
)
_TAPP_destroy_tensor_info = _proto("TAPP_destroy_tensor_info", [TAPP_tensor_info], ctypes.c_int)

_TAPP_create_tensor_product = _proto(
    "TAPP_create_tensor_product",
    [
        ctypes.POINTER(TAPP_tensor_product),
        TAPP_handle,
        ctypes.c_int, TAPP_tensor_info, ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, TAPP_tensor_info, ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, TAPP_tensor_info, ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int, TAPP_tensor_info, ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int,
    ],
    ctypes.c_int,
)
_TAPP_destroy_tensor_product = _proto("TAPP_destroy_tensor_product", [TAPP_tensor_product], ctypes.c_int)

_TAPP_execute_product = _proto(
    "TAPP_execute_product",
    [
        TAPP_tensor_product,
        TAPP_executor,
        ctypes.POINTER(TAPP_status),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ],
    ctypes.c_int,
)
_TAPP_check_success = _proto("TAPP_check_success", [ctypes.c_int], ctypes.c_bool)
_TAPP_explain_error = _proto("TAPP_explain_error", [ctypes.c_int, ctypes.c_size_t, ctypes.c_char_p], ctypes.c_size_t)


def _check(error):
    if _TAPP_check_success(error):
        return error
    length = _TAPP_explain_error(error, 0, None)
    buf = ctypes.create_string_buffer(length + 1)
    _TAPP_explain_error(error, length + 1, buf)
    raise TAPPError(buf.value.decode())


def create_handle():
    handle = TAPP_handle()
    _check(_TAPP_create_handle(ctypes.byref(handle)))
    return handle.value


def destroy_handle(handle):
    _check(_TAPP_destroy_handle(handle))


def create_executor():
    executor = TAPP_executor()
    _check(_TAPP_create_executor(ctypes.byref(executor)))
    return executor.value


def destroy_executor(executor):
    _check(_TAPP_destroy_executor(executor))


def create_tensor_info(datatype, extents, strides):
    nmode = len(extents)
    extents_arr = (ctypes.c_int64 * nmode)(*extents)
    strides_arr = (ctypes.c_int64 * nmode)(*strides)
    info = TAPP_tensor_info()
    _check(_TAPP_create_tensor_info(ctypes.byref(info), datatype, nmode, extents_arr, strides_arr))
    return info.value


def destroy_tensor_info(info):
    _check(_TAPP_destroy_tensor_info(info))


def indices_to_array(indices):
    """Convert a string/iterable of single-character index labels (e.g. "abc")
    into the int64 char-code array TAPP_create_tensor_product expects."""
    codes = [ord(c) for c in indices]
    return (ctypes.c_int64 * len(codes))(*codes)


def create_tensor_product(handle, op_A, A, idx_A, op_B, B, idx_B, op_C, C, idx_C, op_D, D, idx_D, prec):
    plan = TAPP_tensor_product()
    _check(
        _TAPP_create_tensor_product(
            ctypes.byref(plan),
            handle,
            op_A, A, indices_to_array(idx_A),
            op_B, B, indices_to_array(idx_B),
            op_C, C, indices_to_array(idx_C),
            op_D, D, indices_to_array(idx_D),
            prec,
        )
    )
    return plan.value


def destroy_tensor_product(plan):
    _check(_TAPP_destroy_tensor_product(plan))


def execute_product(plan, executor, alpha, A, B, beta, C, D):
    """alpha/beta/A/B/C/D must be ctypes-compatible buffers/pointers (e.g. a
    (c_float * n)() array, or an ndarray's .ctypes.data_as(c_void_p))."""
    # TAPP_destroy_status is declared in the standard (status.h) but has no symbol
    # in libtapp-reference -- no caller in this codebase (C or C++) calls it either.
    # Meaning no formally defined behavior, so the status handle here is
    # intentionally left unfreed, matching the current usage.
    status = TAPP_status()
    _check(_TAPP_execute_product(plan, executor, ctypes.byref(status), alpha, A, B, beta, C, D))
