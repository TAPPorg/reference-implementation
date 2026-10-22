#ifndef TAPP_CUTENSOR_INTERNAL_H_
#define TAPP_CUTENSOR_INTERNAL_H_

/*
 * Internal definitions shared by the source files of the cuTENSOR implementation of TAPP: the objects behind the
 * handles, the error codes and helpers for translating between TAPP and cuTENSOR. Not part of the public interface,
 * that is tapp_cutensor.h.
 */

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
    case TAPP_ERR_INVALID_KEY:      return "Invalid attribute key.";
    case TAPP_ERR_B_C_MISMATCH:     return "B and C must have the same extents, strides and indices.";
    case TAPP_ERR_C_D_MISMATCH:     return "C and D must have the same extents, strides and indices.";
    case TAPP_ERR_UNSUPPORTED_PRECISION: return "The precision needs scalars of another type than the tensors, which is not supported.";
    case TAPP_ERR_MISALIGNED:       return "Tensor data must be aligned to 128 bytes.";
    default:                        return "Unknown TAPP error code.";
    }
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Datatypes
 * ------------------------------------------------------------------------------------------------------------------ */

cutensorDataType_t translate_datatype(TAPP_datatype type);

cutensorComputeDescriptor_t translate_prectype(TAPP_prectype prec, TAPP_datatype datatype);

size_t sizeof_datatype(TAPP_datatype type);

/* ---------------------------------------------------------------------------------------------------------------------
 * Utilities
 * ------------------------------------------------------------------------------------------------------------------ */

const uint32_t kAlignment = 128;
int64_t compute_index(const int64_t* coordinates, int nmode, const int64_t* strides);
void increment_coordinates(int64_t* coordinates, int nmode, const int64_t* extents);
cutensorOperator_t translate_operator(TAPP_element_op op);

// True if the scalar of type `type` at `value` is zero. False for types it does not know
bool is_zero_scalar(TAPP_datatype type, const void* value);

// Copies the scalar of type `type` at `value` to `out`, conjugated if `conj`. out must have room for a TAPP_C64
void load_scalar(TAPP_datatype type, const void* value, bool conj, void* out);

// True if p is NULL or aligned to kAlignment, which the tensor descriptors promise cuTENSOR for the data
bool is_aligned(const void* p);

/* ---------------------------------------------------------------------------------------------------------------------
 * Tensors
 * ------------------------------------------------------------------------------------------------------------------ */

struct tensor_info
{
    int nmode;
    int64_t *extents;
    int64_t *strides;
    size_t elements;
    size_t copy_size;
    int64_t data_offset;
    TAPP_datatype type;
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
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Permutes
 * ------------------------------------------------------------------------------------------------------------------ */

struct permute_plan
{
    cutensorPlan_t plan;       // C = alpha op_A'(A) + beta op_B'(B), op_C folded into op_A' and op_B'
    TAPP_datatype type_C;
    bool conj_scalars;         // op_C is CONJUGATE: alpha and beta are conjugated when executed
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Products
 * ------------------------------------------------------------------------------------------------------------------ */

struct product_plan
{
    cutensorPlan_t plan;
    TAPP_datatype type_D;
    bool conj_scalars;
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
