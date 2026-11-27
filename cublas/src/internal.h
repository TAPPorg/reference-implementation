#ifndef TAPP_CUBLAS_INTERNAL_H_
#define TAPP_CUBLAS_INTERNAL_H_

/*
 * Internal definitions shared by the source files of the cuBLAS/TTGT implementation of TAPP: the objects behind the
 * handles, the error helpers and the datatype translations. Not part of the public interface, that is tapp_cublas.h.
 */

#include <cublasLt.h>
#include <cuda_runtime.h>
//#include <cutt.h>
#include <cutensor.h>
#include <cutensor/types.h>
#include <library_types.h>
#include <map>
#include <tapp.h>
#include <tapp_cublas.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ---------------------------------------------------------------------------------------------------------------------
 * Errors
 * ------------------------------------------------------------------------------------------------------------------ */

enum
{
    TAPP_SUCCESS              = 0,
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
    TAPP_ERR_PRODUCT_PLAN_CREATION_FAILED = 30,
    TAPP_ERR_PRODUCT_PLAN_GEMMUL8_FAILED = 31,
    // The error of a status whose operation has not finished yet, not a failure
    TAPP_ERR_PLAN_WRONG_DEVICE = 32,
    TAPP_ERR_UNSUPPORTED_DATATYPE = 33,
    TAPP_ERR_PLAN_WRONG_EXECUTOR = 34,
    TAPP_ERR_STATUS_RUNNING = 35,
    TAPP_ERR_STATUS_WRONG_DEVICE = 36,
};

typedef enum
{
    TAPP_ERROR_TYPE_TAPP_BASE   = 0,
    TAPP_ERROR_TYPE_CUDA_BASE   = 100000,
    TAPP_ERROR_TYPE_CUBLAS_BASE = 200000,
    // TAPP_ERROR_TYPE_CUTT_BASE   = 300000,
    TAPP_ERROR_TYPE_CUTENSOR_BASE   = 300000,
} TAPP_error_base;

inline TAPP_error tapp_error(cudaError_t e)    { return e == cudaSuccess ? TAPP_SUCCESS : (int)e + TAPP_ERROR_TYPE_CUDA_BASE; }
inline TAPP_error tapp_error(cublasStatus_t e) { return e == CUBLAS_STATUS_SUCCESS ? TAPP_SUCCESS : (int)e + TAPP_ERROR_TYPE_CUBLAS_BASE; }
// inline TAPP_error tapp_error(cuttResult e)     { return e == CUTT_SUCCESS ? TAPP_SUCCESS : (int)e + TAPP_ERROR_TYPE_CUTT_BASE; }
inline TAPP_error tapp_error(cutensorStatus_t e) { return e == CUTENSOR_STATUS_SUCCESS ? TAPP_SUCCESS : (int)e + TAPP_ERROR_TYPE_CUTENSOR_BASE; }

static inline const char* tapp_error_string(int code)
{
    switch (code)
    {
    case TAPP_ERR_EXTENTS_AB:       return "The extents for the indices shared between tensor A and B does not match.";
    case TAPP_ERR_EXTENTS_AD:       return "The extents for the indices shared between tensor A and D does not match.";
    case TAPP_ERR_EXTENTS_BD:       return "The extents for the indices shared between tensor B and D does not match.";
    case TAPP_ERR_D_UNSHARED_IDX:   return "Tensor D has indices not shared with tensor A or B.";
    case TAPP_ERR_CD_NDIM:          return "The tensors C and D have different amount of dimensions.";
    case TAPP_ERR_CD_IDX:           return "The indices of tensor C and D does not line up.";
    case TAPP_ERR_EXTENTS_CD:       return "The extents for the indices shared between tensor C and D does not match.";
    case TAPP_ERR_D_ALIASING:       return "Aliasing found within tensor D.";
    case TAPP_ERR_A_IDX_EXTENTS:    return "An idx in tensor A has two different extents.";
    case TAPP_ERR_B_IDX_EXTENTS:    return "An idx in tensor B has two different extents.";
    case TAPP_ERR_D_IDX_EXTENTS:    return "An idx in tensor D has two different extents.";
    case TAPP_ERR_C_NULL_BETA:      return "C should not be NULL while beta is not zero.";
    case TAPP_ERR_NMODE_NEGATIVE:   return "The number of modes must not be negative.";
    case TAPP_ERR_EXTENTS_NEGATIVE: return "Extents must not be negative.";
    case TAPP_ERR_INVALID_KEY:      return "Invalid attribute key.";
    case TAPP_ERR_NO_EXECUTOR:      return "Executor does not exist.";
    case TAPP_ERR_NOT_IMPLEMENTED:  return "Operation not implemented.";
    case TAPP_ERR_ATTR_CREATION_FAILED: return "The attribute set could not be created.";
    case TAPP_ERR_NULL_POINTER:     return "A required pointer is NULL.";
    case TAPP_ERR_ATTR_INVALID_VALUE: return "The attribute value is invalid for its key.";
    case TAPP_ERR_ATTR_UNKNOWN_KEY: return "The attribute key is not known to this implementation.";
    case TAPP_ERR_ATTR_INVALID_STREAM: return "The stream does not belong to the device given as attribute.";
    case TAPP_ERR_INVALID_DATATYPE: return "The datatype is not supported, or the tensors have different datatypes.";
    case TAPP_ERR_B_C_MISMATCH:     return "B and C must have the same extents, strides and indices.";
    case TAPP_ERR_C_D_MISMATCH:     return "C and D must have the same extents, strides and indices.";
    case TAPP_ERR_UNSUPPORTED_PRECISION: return "The precision is not supported for this datatype.";
    case TAPP_ERR_MISALIGNED:       return "Tensor data must be aligned to 128 bytes.";
    case TAPP_ERR_UNSUPPORTED_ELEMENT_OP: return "The element operation is not supported.";
    case TAPP_ERR_OUT_OF_MEMORY:    return "Out of host memory.";
    case TAPP_ERR_PRODUCT_PLAN_CREATION_FAILED:    return "Product plan creation failed.";
    case TAPP_ERR_STATUS_RUNNING:   return "The operation has not finished yet.";
    case TAPP_ERR_STATUS_WRONG_DEVICE: return "The status is of an operation on another device than the executor.";
    case TAPP_ERR_PLAN_WRONG_DEVICE: return "The plan was created for another device than the executor's.";
    case TAPP_ERR_UNSUPPORTED_DATATYPE: return "Unsupported datatype for the cuBLAS/TTGT back-end.";
    case TAPP_ERR_PLAN_WRONG_EXECUTOR: return "The plan was created with another executor than the one executing it.";
    default:                        return "Unknown TAPP error code.";
    }
}

/* ---------------------------------------------------------------------------------------------------------------------
 * Datatypes
 * ------------------------------------------------------------------------------------------------------------------ */

// Translate a TAPP computational precision into the cuBLAS compute type passed
// to cublasGemmEx. `datatype` resolves TAPP_DEFAULT_PREC.
bool translate_prectype(TAPP_prectype in, TAPP_datatype datatype, cublasComputeType_t& out);

// Translate TAPP precision type for cuTENSOR compute descriptor for cutensorCreateElementwiseBinary. Only used in permute.
bool translate_prectype(TAPP_prectype in, cutensorDataType_t datatype, cutensorComputeDescriptor_t& out);
// Translate TAPP data type into cuTENSOR compute descriptor for permute function. Only used in product.
bool translate_prectype_permute(TAPP_datatype datatype, cutensorComputeDescriptor_t& out);

// If `prec` is one of the variable-precision compute types (TAPP_F_*/TAPP_C_*
// digit counts), return its decimal digit count (> 0); otherwise return 0.
int tapp_prec_digits(TAPP_prectype prec);

// CUDA data type (CUDA_R_32F, ...) for a TAPP storage datatype, for cublasGemmEx.
cudaDataType get_cuda_datatype(TAPP_datatype type);
cutensorDataType_t get_cutensor_datatype(TAPP_datatype type);

size_t sizeof_datatype(TAPP_datatype type);

bool is_64_bit(TAPP_datatype type);

bool is_complex(TAPP_datatype type);

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
    cublasLtHandle_t cublasLt;
    cutensorHandle_t cutensor;
    int device;
    bool owns_stream; // true if TAPP created the stream and must destroy it
    TAPP_emulation_strategy emulation_strategy;
    TAPP_emulation_mantissa_control emulation_mantissa_control;
    void *workspace = nullptr;
#if OZAKI_EMULATION_VERSION==2
    TAPP_emulation_os_II_backend os_II_backend;
#endif
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Products
 * ------------------------------------------------------------------------------------------------------------------ */

// A built tensor-contraction plan for the cuBLAS/TTGT back-end.
//
// The TTGT (transpose-transpose-GEMM-transpose) schedule is computed once in
// TAPP_create_tensor_product and replayed in TAPP_execute_product: each input
// is optionally permuted into a matrix with cuTT, the matrices are multiplied
// with cuBLAS GEMM, and the result is optionally permuted back. Everything the
// execution needs is stored here directly (no separate plan/info objects).

struct product_plan
{
    bool failed;         // set when the contraction is outside TTGT's support

    // Transpose plans. transposeX == false means tensor X is already in the
    // layout the GEMM expects, so it is fed to the GEMM unchanged.
    bool transposeA, transposeB, transposeC;
    // cuttHandle planA, planB, planC;
    cutensorPlan_t planA, planB, planC, planD;

    // GEMM configuration (CT = alpha * op(A) * op(B)).
    cublasOperation_t transa, transb;
    int m, n, k, lda, ldb, ldc;

#if OZAKI_EMULATION_VERSION==2
    // Num of GEMM for Ozaki-II scheme
    int num_gemm;
#else
    cublasLtMatmulDescOpaque_t operationDesc;
    cublasLtMatrixLayoutOpaque_t Adesc, Bdesc, Cdesc;
    cublasLtMatmulHeuristicResult_t heuristicResult;
#endif

    // Element counts, for sizing the transposed device buffers.
    size_t bytes_A, bytes_B, bytes_D;
    TAPP_datatype type_D;

    TAPP_element_op op_D;

    size_t workspaceSize;
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Permutes
 * ------------------------------------------------------------------------------------------------------------------ */

struct permute_plan
{
    cutensorPlan_t plan;       // C = alpha op_A'(A) + beta op_B'(B), op_C folded into op_A' and op_B'
    cutensorDataType_t type_C;
    bool conj_scalars;         // op_C is CONJUGATE: alpha and beta are conjugated when executed
    // Of the executor the plan was created with; it can only run there
    int device;
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

/* ---------------------------------------------------------------------------------------------------------------------
 * Utilities
 * ------------------------------------------------------------------------------------------------------------------ */

const uint32_t ALIGNMENT_REQUIREMENT = 128;
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

// Owns a cuTENSOR object and destroys it when it goes out of scope, unless release() handed it on
template <typename T, cutensorStatus_t (*Destroy)(T)>
struct cutensor_owner
{
    T handle = nullptr;

    cutensor_owner() = default;
    ~cutensor_owner() { if (handle) Destroy(handle); }
    cutensor_owner(const cutensor_owner&) = delete;
    cutensor_owner& operator=(const cutensor_owner&) = delete;

    // Gives up ownership, for example to a plan struct that destroys it later
    T release() { T h = handle; handle = nullptr; return h; }
};

typedef cutensor_owner<cutensorTensorDescriptor_t, cutensorDestroyTensorDescriptor> tensor_descriptor;
typedef cutensor_owner<cutensorOperationDescriptor_t, cutensorDestroyOperationDescriptor> operation_descriptor;
typedef cutensor_owner<cutensorPlanPreference_t, cutensorDestroyPlanPreference> plan_preference;
typedef cutensor_owner<cutensorPlan_t, cutensorDestroyPlan> plan_owner;

/* ---------------------------------------------------------------------------------------------------------------------
 * Tensors
 * ------------------------------------------------------------------------------------------------------------------ */

// Logical/physical description of a dense tensor. The TTGT back-end works on
// dense (generalized column-major) layouts; `extents`/`strides` are retained
// both for the host<->device copy bounds and to feed the TTGT plan.
struct tensor_info
{
    int nmode;
    int64_t* extents;
    int64_t* strides;
    size_t elements;
    TAPP_datatype type;
};

// True if A and B have the same extents, strides and indices, so one can be computed in the memory layout of the other
bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B);
cutensorStatus_t create_descriptor(cutensorHandle_t handle, const tensor_info* info, tensor_descriptor& desc);

#endif  /* TAPP_CUBLAS_INTERNAL_H_ */
