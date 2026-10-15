#ifndef TAPP_TBLIS_INTERNAL_H_
#define TAPP_TBLIS_INTERNAL_H_

/*
 * Internal definitions shared by the source files of the TBLIS implementation of TAPP: the objects behind the handles,
 * the error codes and helpers for translating between TAPP and TBLIS. Not part of the public interface, that is
 * tapp_tblis.h.
 */

#include <tapp.h>
#include <tapp_tblis.h>

#pragma GCC diagnostic push
#include <tblis.h>
#pragma GCC diagnostic pop

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

// CPU masks (TAPP_CPU_MASK) use the Linux affinity API (cpu_set_t, sched_setaffinity, pthread_setaffinity_np). On other
// platforms the attribute is refused and executors run on any CPU
#if defined(__linux__)
#  define TAPP_TBLIS_HAS_CPU_MASK 1
#  include <pthread.h>
#  include <sched.h>
#endif

/* ---------------------------------------------------------------------------------------------------------------------
 * Errors
 * ------------------------------------------------------------------------------------------------------------------ */

enum TAPP_TBLIS_error : TAPP_error
{
    TAPP_TBLIS_SUCCESS = 0,
    TAPP_TBLIS_ERROR_NULL_POINTER,
    TAPP_TBLIS_ERROR_NEGATIVE_NMODE,
    TAPP_TBLIS_ERROR_NEGATIVE_EXTENT,
    TAPP_TBLIS_ERROR_UNSUPPORTED_DATATYPE,
    TAPP_TBLIS_ERROR_OUT_OF_MEMORY,
    TAPP_TBLIS_ERROR_DATATYPE_MISMATCH,
    TAPP_TBLIS_ERROR_EXTENT_MISMATCH,
    TAPP_TBLIS_ERROR_C_D_MISMATCH,
    TAPP_TBLIS_ERROR_INDEX_NOT_IN_INPUTS,
    TAPP_TBLIS_ERROR_TOO_MANY_INDICES,
    TAPP_TBLIS_ERROR_UNSUPPORTED_CONTRACTION,
    TAPP_TBLIS_ERROR_UNSUPPORTED_ELEMENT_OP,
    TAPP_TBLIS_ERROR_UNSUPPORTED_PRECISION,
    TAPP_TBLIS_ERROR_UNKNOWN_ATTR_KEY,
    TAPP_TBLIS_ERROR_INVALID_ATTR_VALUE,
    TAPP_TBLIS_ERROR_CPU_MASK_UNAVAILABLE,
    TAPP_TBLIS_ERROR_INVALID_STATUS,
};

/* ---------------------------------------------------------------------------------------------------------------------
 * Datatypes
 * ------------------------------------------------------------------------------------------------------------------ */

/*
 * Convert a TAPP storage datatype to the corresponding TBLIS type.
 * Returns false if the datatype has no TBLIS equivalent (e.g. TAPP_F16,
 * TAPP_BF16), in which case out is left unchanged.
 *
 * Note: the numeric values differ between the two libraries
 * (TAPP_F64 = 1 but TYPE_DOUBLE = 2, TAPP_C32 = 2 but TYPE_SCOMPLEX = 1),
 * so the values must never be cast directly.
 */
inline bool to_tblis_type(TAPP_datatype in, tblis::type_t& out)
{
    switch (in)
    {
        case TAPP_F32: out = tblis::TYPE_FLOAT;    return true;
        case TAPP_F64: out = tblis::TYPE_DOUBLE;   return true;
        case TAPP_C32: out = tblis::TYPE_SCOMPLEX; return true;
        case TAPP_C64: out = tblis::TYPE_DCOMPLEX; return true;
        default:                                   return false;
    }
}

/*
 * Size in bytes of one element of the given TBLIS type, 0 if the type is unknown.
 */
inline size_t tblis_type_size(tblis::type_t type)
{
    switch (type)
    {
        case tblis::TYPE_FLOAT:    return sizeof(float);
        case tblis::TYPE_DOUBLE:   return sizeof(double);
        case tblis::TYPE_SCOMPLEX: return sizeof(tblis::scomplex);
        case tblis::TYPE_DCOMPLEX: return sizeof(tblis::dcomplex);
        default:                   return 0;
    }
}

/*
 * Read a scalar of the given TBLIS type from untyped memory, as passed for
 * alpha and beta through the TAPP API. The type must be one returned by
 * to_tblis_type.
 */
inline tblis::tblis_scalar to_tblis_scalar(tblis::type_t type, const void* value)
{
    switch (type)
    {
        case tblis::TYPE_FLOAT:    return tblis::tblis_scalar(*static_cast<const float*>(value));
        case tblis::TYPE_DOUBLE:   return tblis::tblis_scalar(*static_cast<const double*>(value));
        case tblis::TYPE_SCOMPLEX: return tblis::tblis_scalar(*static_cast<const tblis::scomplex*>(value));
        case tblis::TYPE_DCOMPLEX: return tblis::tblis_scalar(*static_cast<const tblis::dcomplex*>(value));
        default:                   return tblis::tblis_scalar(0.0);  // unreachable for valid types, tblis_scalar has no default constructor
    }
}

/* ---------------------------------------------------------------------------------------------------------------------
 * TBLIS utilities
 * ------------------------------------------------------------------------------------------------------------------ */

// Maps a TAPP storage datatype to the corresponding TBLIS type, returns false if unsupported (e.g. F16, BF16)
bool translate_datatype_to_tblis(TAPP_datatype type, tblis::type_t* tblis_type);

// Maps TAPP index labels to TBLIS labels (char). label_map must be shared between all tensors of one product so that
// equal TAPP labels get equal TBLIS labels. Returns false if the product uses more distinct labels than TBLIS supports.
bool translate_idx_to_tblis(int nmode,
                            const int64_t* idx,
                            std::unordered_map<int64_t, tblis::label_type>& label_map,
                            std::vector<tblis::label_type>& tblis_idx);

// Wraps raw data in a tblis_tensor, optionally scaled (alpha/beta) and conjugated (op)
void init_tblis_tensor(tblis::tblis_tensor* tensor,
                       TAPP_datatype type,
                       int nmode,
                       const tblis::len_type* extents,
                       const tblis::stride_type* strides,
                       const void* data,
                       TAPP_element_op op,
                       const void* scalar);

/* ---------------------------------------------------------------------------------------------------------------------
 * Tensors
 * ------------------------------------------------------------------------------------------------------------------ */

struct tensor_info
{
    tblis::type_t type;
    std::vector<tblis::len_type> extents;
    std::vector<tblis::stride_type>  strides;
    int conj;
};

// Creates a tblis_tensor that points to data and to the shape of info: nothing is copied, so info must outlive it.
// scalar and conj replace 1 and the conj of info when they are not null.
tblis::tblis_tensor create_tblis_tensor(const tensor_info& info,
                                        const void* data,
                                        const tblis::tblis_scalar* scalar,
                                        const bool* conj);

/* ---------------------------------------------------------------------------------------------------------------------
 * Attributes
 * ------------------------------------------------------------------------------------------------------------------ */

typedef std::map<TAPP_key, std::vector<char>> attributes;

/* ---------------------------------------------------------------------------------------------------------------------
 * Executors
 * ------------------------------------------------------------------------------------------------------------------ */

struct executor
{
    std::thread worker;
    std::queue<std::function<TAPP_error()>> tasks;

    std::mutex m;
    std::condition_variable cv;
    std::condition_variable done_cv;

    uint64_t completed = 0;
    uint64_t submitted = 0;

    bool stopping = false;

    unsigned thread_count;
#ifdef TAPP_TBLIS_HAS_CPU_MASK
    cpu_set_t cpu_mask;
#endif
    bool has_cpu_mask;
};

uint64_t submit_task(executor* exec, std::function<TAPP_error()> task);

/* ---------------------------------------------------------------------------------------------------------------------
 * Products
 * ------------------------------------------------------------------------------------------------------------------ */

struct product_plan
{
    tensor_info A;
    tensor_info B;
    tensor_info C;
    int64_t size_D;
    bool conj_scalars;
    std::vector<tblis::label_type> idx_A;
    std::vector<tblis::label_type> idx_B;
    std::vector<tblis::label_type> idx_C;
    tblis::type_t scalar_type;
};

bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B);

TAPP_error execute_product(const product_plan* plan,
                           const executor* exec,
                           const tblis::tblis_scalar alpha,
                           const void* A,
                           const void* B,
                           const tblis::tblis_scalar beta,
                           const void* C,
                                 void* D);

#endif  /* TAPP_TBLIS_INTERNAL_H_ */
