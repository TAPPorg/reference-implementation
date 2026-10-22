#ifndef TAPP_CUTENSOR_INTERNAL_H_
#define TAPP_CUTENSOR_INTERNAL_H_

/*
 * Internal definitions shared by the source files of the cuTENSOR implementation of TAPP: the objects behind the
 * handles, the error codes and helpers for translating between TAPP and cuTENSOR. Not part of the public interface,
 * that is tapp_cutensor.h.
 */

#include <cutensor/types.h>
#include <tapp.h>
#include <tapp_cutensor.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

/* ---------------------------------------------------------------------------------------------------------------------
 * Errors
 * ------------------------------------------------------------------------------------------------------------------ */

enum
{
    TAPP_SUCCESS          = 0,
    TAPP_ERR_EXTENTS_AB       = 1,
    TAPP_ERR_EXTENTS_AD       = 2,
    TAPP_ERR_EXTENTS_BD       = 3,
    TAPP_ERR_D_UNSHARED_IDX   = 4,
    TAPP_ERR_CD_NDIM          = 5,
    TAPP_ERR_CD_IDX           = 6,
    TAPP_ERR_EXTENTS_CD       = 7,
    TAPP_ERR_D_ALIASING       = 8,
    TAPP_ERR_A_IDX_EXTENTS    = 9,
    TAPP_ERR_B_IDX_EXTENTS    = 10,
    TAPP_ERR_D_IDX_EXTENTS    = 11,
    TAPP_ERR_C_NULL_BETA      = 12,
    TAPP_ERR_NMODE_NEGATIVE   = 13,
    TAPP_ERR_EXTENTS_NEGATIVE = 14,
    TAPP_ERR_INVALID_KEY      = 15,
    TAPP_ERR_NO_EXECUTOR      = 16,
    TAPP_ERR_NOT_IMPLEMENTED  = 17,
    TAPP_ERR_ATTR_CREATION_FAILED  = 18,
    TAPP_ERR_NULL_POINTER  = 19,
    TAPP_ERR_ATTR_INVALID_VALUE  = 20,
    TAPP_ERR_ATTR_UNKNOWN_KEY = 21,
    TAPP_ERR_ATTR_INVALID_STREAM = 22,
    TAPP_ERR_INVALID_DATATYPE = 23,
    TAPP_ERR_B_C_MISMATCH = 24,
    TAPP_ERR_C_D_MISMATCH = 25,
    TAPP_ERR_UNSUPPORTED_PRECISION = 26,
    TAPP_ERR_MISALIGNED = 27,
    TAPP_ERR_UNSUPPORTED_ELEMENT_OP = 28,
    TAPP_ERR_OUT_OF_MEMORY = 29,
    // The error of a status whose operation has not finished yet, not a failure
    TAPP_ERR_STATUS_RUNNING = 30,
    TAPP_ERR_STATUS_WRONG_DEVICE = 31,
};

typedef enum
{
    TAPP_ERROR_TYPE_TAPP_BASE  = 0,
    TAPP_ERROR_TYPE_CUDA_BASE  = 100000,
    TAPP_ERROR_TYPE_CUTENSOR_BASE = 200000,
} TAPP_error_base;

inline TAPP_error tapp_error(cudaError_t e)      {
    return e == cudaSuccess ? TAPP_SUCCESS : (int) e+TAPP_ERROR_TYPE_CUDA_BASE;
}
inline TAPP_error tapp_error(cutensorStatus_t e) { return e == CUTENSOR_STATUS_SUCCESS ? TAPP_SUCCESS : (int) e + TAPP_ERROR_TYPE_CUTENSOR_BASE; }

/* Message for a TAPP_ERROR_TYPE_TAPP code, shared so every implementation
 * reports these errors identically. */
static inline const char* tapp_error_string(int code)
{
    switch (code)
    {
    case TAPP_ERR_NMODE_NEGATIVE:   return "The number of modes must not be negative.";
    case TAPP_ERR_EXTENTS_NEGATIVE: return "Extents must not be negative.";
    case TAPP_ERR_INVALID_KEY:      return "Invalid attribute key.";
    case TAPP_ERR_NULL_POINTER:     return "A required pointer is NULL.";
    case TAPP_ERR_INVALID_DATATYPE: return "The datatype is not supported, or the tensors have different datatypes.";
    case TAPP_ERR_B_C_MISMATCH:     return "B and C must have the same extents, strides and indices.";
    case TAPP_ERR_C_D_MISMATCH:     return "C and D must have the same extents, strides and indices.";
    case TAPP_ERR_UNSUPPORTED_PRECISION: return "The precision is not supported for this datatype.";
    case TAPP_ERR_MISALIGNED:       return "Tensor data must be aligned to 128 bytes.";
    case TAPP_ERR_UNSUPPORTED_ELEMENT_OP: return "The element operation is not supported.";
    case TAPP_ERR_OUT_OF_MEMORY:    return "Out of host memory.";
    case TAPP_ERR_STATUS_RUNNING:   return "The operation has not finished yet.";
    case TAPP_ERR_STATUS_WRONG_DEVICE: return "The status is of an operation on another device than the executor.";
    default:                        return "Unknown TAPP error code.";
    }
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Datatypes
 * ------------------------------------------------------------------------------------------------------------------ */

bool translate_datatype(TAPP_datatype in, cutensorDataType_t& out);

bool translate_prectype(TAPP_prectype in, cutensorDataType_t datatype, cutensorComputeDescriptor_t& out);

size_t sizeof_datatype(cutensorDataType_t type);

/* ---------------------------------------------------------------------------------------------------------------------
 * Utilities
 * ------------------------------------------------------------------------------------------------------------------ */

const uint32_t kAlignment = 128;
bool translate_operator(TAPP_element_op in, cutensorOperator_t& out);

// True if op is an element operation this implementation supports
bool is_valid_op(TAPP_element_op op);

// True if the scalar of type `type` at `value` is zero. False for types it does not know
bool is_zero_scalar(cutensorDataType_t type, const void* value);

// Copies the scalar of type `type` at `value` to `out`, conjugated if `conj`. out must have room for a TAPP_C64
void load_scalar(cutensorDataType_t type, const void* value, bool conj, void* out);

// True if p is NULL or aligned to kAlignment, which the tensor descriptors promise cuTENSOR for the data
bool is_aligned(const void* p);

// Makes a device current for as long as it exists and then makes the device current again that was current before, so
// TAPP functions do not change the current device of the calling program. Check error before using the device
struct device_scope
{
    explicit device_scope(int device)
    {
        if (cudaGetDevice(&previous) != cudaSuccess) previous = -1;
        error = cudaSetDevice(device);
    }

    ~device_scope()
    {
        if (previous >= 0) cudaSetDevice(previous);
    }

    device_scope(const device_scope&) = delete;
    device_scope& operator=(const device_scope&) = delete;

    int previous;      // -1 if it could not be determined, then nothing is restored
    cudaError_t error; // Of making the device current
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Tensors
 * ------------------------------------------------------------------------------------------------------------------ */

struct tensor_info
{
    cutensorDataType_t type;
    std::vector<int64_t> extents;
    std::vector<int64_t> strides;
};

// True if A and B have the same extents, strides and indices, so one can be computed in the memory layout of the other
bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B);

/* ---------------------------------------------------------------------------------------------------------------------
 * Attributes
 * ------------------------------------------------------------------------------------------------------------------ */

typedef std::map<TAPP_key, std::vector<char>> attributes;

/* ---------------------------------------------------------------------------------------------------------------------
 * Executors
 * ------------------------------------------------------------------------------------------------------------------ */

struct executor
{
    cudaStream_t stream;
    cutensorHandle_t libhandle;
    int device;
    bool owns_stream; // true if TAPP created the stream and must destroy it
    // Reused by the products of this executor: they run one after another on its stream. Not safe to use one executor
    // from several host threads at the same time
    void *workspace = nullptr;
    uint64_t workspace_capacity = 0;
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Permutes
 * ------------------------------------------------------------------------------------------------------------------ */

struct permute_plan
{
    cutensorPlan_t plan;       // C = alpha op_A'(A) + beta op_B'(B), op_C folded into op_A' and op_B'
    cutensorDataType_t type_C;
    bool conj_scalars;         // op_C is CONJUGATE: alpha and beta are conjugated when executed
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Products
 * ------------------------------------------------------------------------------------------------------------------ */

struct product_plan
{
    cutensorPlan_t plan;
    cutensorDataType_t type_D;
    bool conj_scalars;
    uint64_t workspace_size = 0;
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Status
 * ------------------------------------------------------------------------------------------------------------------ */

struct status
{
    cudaEvent_t event;
    int device;
};

// Allocate a status object recording the current point of `stream`.
TAPP_error create_status(cudaStream_t stream, int device, TAPP_status* status);

#endif  /* TAPP_CUTENSOR_INTERNAL_H_ */
