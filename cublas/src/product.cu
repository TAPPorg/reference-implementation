#include "internal.h"
#include "tapp/executor.h"

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#if OZAKI_EMULATION_VERSION==2
#include <gemmul8.hpp>
#include <complex>
#include <gemmul8.hpp>
#include <type_traits>
#include <variant>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/* The number of GEMM in OS-II is related to the number of GEMM according to recommandations in README this repo: 
* https://github.com/RIKEN-RCCS/GEMMul8
* 30b5b5d01e96c0e4a490396d5a806e86bcc36581
* formatted as effective bits / number of low-precision real matrix multiplications.
*
* num_moduli 	INT8-Real 	INT8-Complex 	FP8-Real 	FP8-Complex
* 2 	          7 / 2 	   7 / 4 	    13 / 5 	     12 / 10
* 3 	          11 / 3 	   11 / 6 	    18 / 7 	     16 / 15
* 4 	          15 / 4 	   15 / 8 	    23 / 9 	     22 / 21
* 5 	          19 / 5 	   19 / 10 	    28 / 11 	  27 / 28
* 6 	          23 / 6 	   23 / 12 	    34 / 14 	  32 / 34
* 7 	          27 / 7 	   27 / 14 	    39 / 17 	  35 / 37
* 8 	          31 / 8 	   30 / 16 	    44 / 20 	  40 / 43
* 9 	          35 / 9 	   34 / 18 	    48 / 22 	  46 / 49
* 10 	          39 / 10 	   38 / 20 	    53 / 25 	  51 / 55
* 11 	          43 / 11 	   41 / 22 	    58 / 28 	  56 / 61
* 12 	          47 / 12 	   45 / 24 	    63 / 31 	  61 / 67
* 13 	          51 / 13 	   48 / 26 	    68 / 34 	  66 / 73
* 14 	          55 / 14 	   52 / 28 	    73 / 37 	  70 / 78
* 15 	          58 / 15 	   55 / 30 	    78 / 40 	  75 / 84
* 16 	          62 / 16 	   58 / 32 	    84 / 43 	  80 / 90
* 17 	          66 / 17 	   62 / 34 	    89 / 46 	  85 / 96
* 18 	          70 / 18 	   65 / 36 	    94 / 49 	  90 / 102
* 19 	          73 / 19 	   68 / 38 	    99 / 52 	  95 / 108
* 20 	          77 / 20 	   71 / 40 	    102 / 54 	  100 / 114
*/

#if OZAKI_EMULATION_VERSION==2
std::map<int,int> OS_II_INT8_table = {
   {2 , 7 }, 
   {3 , 11}, 
   {4 , 15}, 
   {5 , 19}, 
   {6 , 23}, 
   {7 , 27}, 
   {8 , 30}, 
   {9 , 34}, 
   {10, 38}, 
   {11, 41}, 
   {12, 45}, 
   {13, 48}, 
   {14, 52}, 
   {15, 55}, 
   {16, 58}, 
   {17, 62}, 
   {18, 65}, 
   {19, 68}, 
   {20, 71}, 
};
#endif

// ===========================================================================
// CUDA / cuBLAS / cuTT error checks. Unlike the upstream my-ttgt macros these throw
// instead of exit()-ing, so the binding can report a TAPP error and keep the
// host process alive.
// ===========================================================================

// Carries the TAPP error code of the failed call to the catch in TAPP_execute_product
// struct tapp_exception : std::runtime_error
// {
//     TAPP_error code;
//     tapp_exception(TAPP_error code, const std::string& what) : std::runtime_error(what), code(code) {}
// };
// 
// static inline void cuda_check(cudaError_t code)
// {
//     if (code != cudaSuccess)
//         throw tapp_exception(tapp_error(code), std::string("CUDA error: ") + cudaGetErrorString(code));
// }
// 
// static inline void cublas_check(cublasStatus_t stat)
// {
//     if (stat != CUBLAS_STATUS_SUCCESS)
//         throw tapp_exception(tapp_error(stat), std::string("cuBLAS error: ") + cublasGetStatusString(stat));
// }
// 
// static inline void cutt_check(cuttResult err)
// {
//     if (err != CUTT_SUCCESS)
//         throw tapp_exception(tapp_error(err), "cuTT error " + std::to_string((int)err));
// }

// ===========================================================================
// Index/permutation helpers implementing the TTGT index bookkeeping.
// ===========================================================================

// Contracted ("bounded") labels of `mode`: those not present in `modeC`.
static void get_bounded_indices(int64_t** bounded, int* n_bounded,
                                const int64_t* mode, int rank,
                                const int64_t* modeC, int rankC)
{
    std::vector<int64_t> tmp(rank);
    *n_bounded = 0;
    for (int i = 0; i < rank; i++)
    {
        int64_t idx = mode[i];
        if (std::find(modeC, modeC + rankC, idx) == modeC + rankC)
            tmp[(*n_bounded)++] = idx;
    }
    *bounded = new int64_t[*n_bounded > 0 ? *n_bounded : 1];
    for (int i = 0; i < *n_bounded; i++)
        (*bounded)[i] = tmp[i];
}

// Free labels of `mode`: those also present in `modeC` (first n_free of them).
static void get_free_indices(int64_t* free_idx, int n_free,
                             const int64_t* mode, int rank,
                             const int64_t* modeC, int rankC)
{
    int index = 0;
    for (int i = 0; i < rank && index < n_free; i++)
    {
        int64_t idx = mode[i];
        if (std::find(modeC, modeC + rankC, idx) != modeC + rankC)
            free_idx[index++] = idx;
    }
}

// permutation[i] = position, in `mode`, of the label sitting at modeT[i].
static void get_permutation(int* permutation, const int64_t* mode,
                            const int64_t* modeT, int rank)
{
    std::unordered_map<int64_t, int> pos;
    for (int i = 0; i < rank; i++)
        pos[modeT[i]] = i;
    for (int i = 0; i < rank; i++)
        permutation[i] = pos[mode[i]];
}

// Prepend a value to a dim array (complex64 cuTT workaround). Caller deletes[].
static int* prepend(const int* a, int rank, int value)
{
    int* out = new int[rank + 1];
    out[0] = value;
    std::memcpy(out + 1, a, rank * sizeof(int));
    return out;
}

// Shift a permutation by one and place axis 0 first (complex64 workaround).
static int* shift_permutation(const int* perm, int size)
{
    int* out = new int[size + 1];
    out[0] = 0;
    for (int i = 0; i < size; i++)
        out[i + 1] = perm[i] + 1;
    return out;
}

// Fill `out` (>= 16 bytes) with `value` cast to `type`. Used for the GEMM beta
// (0) and the GEAM scale (1).
static void make_scalar(TAPP_datatype type, double value, void* out)
{
    switch (type)
    {
    case TAPP_F32: *(float*)out = (float)value; break;
    case TAPP_F64: *(double*)out = value; break;
    case TAPP_C32: ((float*)out)[0] = (float)value; ((float*)out)[1] = 0.0f; break;
    case TAPP_C64: ((double*)out)[0] = value; ((double*)out)[1] = 0.0; break;
    default: *(float*)out = (float)value; break;
    }
}

// ===========================================================================
// Create cuTENSOR plans for A -> AT and B -> BT. Returns a value for p->failed, so false if it succeed.
// ===========================================================================
static cutensorStatus_t create_cutensor_permute_plan(cutensorHandle_t cutensor, cutensorPlan_t* plan, tensor_info* A_info, int64_t* modeA, int64_t* modeAT, TAPP_element_op op_A,
                               const int* permutation)
{

    uint32_t nmodeA = A_info->nmode;
    cutensorStatus_t status;
    int64_t extentsAT[nmodeA];
    int64_t stridesAT[nmodeA];                                                     
    cutensorOperator_t op_cutensor;
    cutensorComputeDescriptor_t cutensorprec;

    std::vector<int32_t> cuidx_A = std::vector<int32_t>(modeA, modeA + nmodeA);
    std::vector<int32_t> cuidx_AT = std::vector<int32_t>(modeAT, modeAT + nmodeA);

    // Tensor descriptors
    cutensorTensorDescriptor_t descA;
    status = cutensorCreateTensorDescriptor(
        cutensor,
        &descA,
        nmodeA,
        A_info->extents,
        A_info->strides,
        get_cuda_datatype(A_info->type),
        ALIGNMENT_REQUIREMENT
    );
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    for(uint32_t i=0; i<nmodeA; i++) extentsAT[i] = A_info->extents[permutation[i]];
    stridesAT[0] = 1;
    for(uint32_t i=1; i<nmodeA; i++) stridesAT[i] = stridesAT[i-1] * extentsAT[i-1];

    cutensorTensorDescriptor_t descAT;
    status = cutensorCreateTensorDescriptor(
        cutensor,
        &descAT,
        nmodeA,
        extentsAT,
        stridesAT,
        get_cuda_datatype(A_info->type),
        ALIGNMENT_REQUIREMENT
    );
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // Permute descriptor and plan
    cutensorOperationDescriptor_t descPermute;
    translate_operator(op_A, op_cutensor);
    if(!translate_prectype_permute(A_info->type, cutensorprec)){
        status = CUTENSOR_STATUS_NOT_SUPPORTED;
        goto cleanup;
    }
    status = cutensorCreatePermutation(
        cutensor,
        &descPermute,
        descA,
        cuidx_A.data(),
        op_cutensor,
        descAT,
        cuidx_AT.data(),
        cutensorprec
    );
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    cutensorPlanPreference_t pref;
    status = cutensorCreatePlanPreference(cutensor,
                &pref,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    status = cutensorCreatePlan(cutensor, plan, descPermute, pref, 0);
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // Cleanup
    status = cutensorDestroyTensorDescriptor(descA);
    status = cutensorDestroyTensorDescriptor(descAT);
    status = cutensorDestroyPlanPreference(pref);
    status = cutensorDestroyOperationDescriptor(descPermute);

    return CUTENSOR_STATUS_SUCCESS;

cleanup:
    if(descA) cutensorDestroyTensorDescriptor(descA);
    if(descAT) cutensorDestroyTensorDescriptor(descAT);
    if(pref) cutensorDestroyPlanPreference(pref);
    if(descPermute) cutensorDestroyOperationDescriptor(descPermute);

    return status;
}

// ===========================================================================
// cuTT plan creation with extent-1 squeezing.
//
// cuTT rejects any transpose whose dims contain a size-1 axis (and rank < 2)
// with CUTT_INVALID_PARAMETER. A size-1 axis carries no data, so it can be
// dropped from (rank, dim, permutation) without changing which bytes move.
// Returns true if a real transpose plan was created, false if the transpose is
// the identity / degenerate (no data movement: the source buffer is already in
// the target layout). On a genuine cuTT failure the (reduced) parameters are
// printed and an exception is thrown.
// ===========================================================================
// static bool cutt_plan_squeezed(cudaStream_t stream, cuttHandle* plan, int rank, const int* dim,
//                                const int* permutation, size_t sizeofType,
//                                const char* which)
// {
//     std::vector<int> old_to_new(rank, -1);
//     std::vector<int> sdim;
//     int new_rank = 0;
//     for (int i = 0; i < rank; i++)
//         if (dim[i] != 1)
//         {
//             old_to_new[i] = new_rank++;
//             sdim.push_back(dim[i]);
//         }
// 
//     std::vector<int> sperm;
//     for (int i = 0; i < rank; i++)
//     {
//         int axis = permutation[i];
//         if (dim[axis] != 1)
//             sperm.push_back(old_to_new[axis]);
//     }
// 
//     bool identity = true;
//     for (int i = 0; i < new_rank; i++)
//         if (sperm[i] != i) { identity = false; break; }
//     if (new_rank < 2 || identity)
//         return false;
// 
//     cuttResult err = cuttPlan(plan, new_rank, sdim.data(), sperm.data(), sizeofType, stream);
//     if (err != CUTT_SUCCESS)
//     {
//         fprintf(stderr,
//                 "TTGT: cuttPlan failed for tensor %s (err=%d): rank=%d "
//                 "elemsize=%zu dim=[",
//                 which, err, new_rank, sizeofType);
//         for (int i = 0; i < new_rank; i++)
//             fprintf(stderr, "%d%s", sdim[i], i + 1 < new_rank ? "," : "");
//         fprintf(stderr, "] perm=[");
//         for (int i = 0; i < new_rank; i++)
//             fprintf(stderr, "%d%s", sperm[i], i + 1 < new_rank ? "," : "");
//         fprintf(stderr, "]\n");
//         throw std::runtime_error("cuTT plan creation failed");
//     }
//     return true;
// }

// ===========================================================================
// Support checks.
// ===========================================================================
static bool is_supported_datatype(TAPP_datatype type)
{
    return type == TAPP_F32 || type == TAPP_F64 ||
           type == TAPP_C32 || type == TAPP_C64;
}

// TTGT assumes a dense, positive-stride layout. A non-positive stride on an
// extent>1 axis is a reversal (negative) or broadcast (zero), neither of which
// TTGT supports; a negative stride would also make the host<->device copy-size
// underflow into a huge allocation. The stride of an extent<=1 axis is ignored.
static bool tensor_strides_supported(const struct tensor_info* t)
{
    for (int i = 0; i < t->nmode; i++)
        if (t->extents[i] > 1 && t->strides[i] <= 0)
            return false;
    return true;
}

static bool scalar_is_nonzero(const void* storage, TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32: return *(const float*)storage != 0.0f;
    case TAPP_F64: return *(const double*)storage != 0.0;
    case TAPP_C32: { const float* v = (const float*)storage; return v[0] != 0.0f || v[1] != 0.0f; }
    case TAPP_C64: { const double* v = (const double*)storage; return v[0] != 0.0 || v[1] != 0.0; }
    default: return true;
    }
}

// ===========================================================================
// Dispatch pattern for interfacing with Ozaki-II project
// The underlying GEMM assumes for simplicity that CUBLAS_OP = CUBLAS_OP_N for A and B.
// During the Nvidia Idris-Hackathlon (2026), we identified cuBLAS GEMM as the bottleneck.
// So the previous assumption is not a big performance loss.
// ===========================================================================
#if OZAKI_EMULATION_VERSION==2
using CudaTypeVar = std::variant<
    std::type_identity<float>,
    std::type_identity<double>,
    std::type_identity<cuComplex>,
    std::type_identity<cuDoubleComplex>
>;

CudaTypeVar get_cpp_datatype(TAPP_datatype dtype) {
    switch (dtype) {
        case TAPP_F32: return std::type_identity<float>{};
        case TAPP_F64: return std::type_identity<double>{};
        case TAPP_C32: return std::type_identity<cuComplex>{};
        case TAPP_C64: return std::type_identity<cuDoubleComplex>{};
    }
    throw std::runtime_error("Unsuported value");
}

template <gemmul8::Backend BACKEND>
void dispatch_gemmul8(
           cublasHandle_t cublas,
           CudaTypeVar typeC, 
           cublasOperation_t transa,
           cublasOperation_t transb,
           size_t m, 
           size_t n, 
           size_t k,
           const void* alpha, 
           const void* A_d, 
           size_t lda,
           const void* B_d, 
           size_t ldb,
           const void* zero, 
           void* devCT, 
           size_t ldc,
           int num_moduli, 
           bool fastmode, 
           void* work_rem, 
           void* workA, 
           void* workB,
           bool enable_skip_scalA, 
           bool enable_skip_scalB, 
           bool skip_scalA, 
           bool skip_scalB)
{
    std::visit([&]<typename typeC>(
      std::type_identity<typeC>) {
// Ozaki-II doesnt not have support for FP8 on H100
#if OZAKI_II_COMPUTE_CAPABILITY>1000000
         if constexpr(true) gemmul8::gemm<typeC, BACKEND>(
           cublas,
           transa, transb,
           m, n, k,
           (typeC*) alpha, (typeC*) A_d, lda,
           (typeC*) B_d, ldb, (typeC*) zero,
           (typeC*) devCT, ldc,
           num_moduli,
           fastmode, work_rem, workA, workB, enable_skip_scalA, enable_skip_scalB,
           skip_scalA, skip_scalB
        );
#else
         if (BACKEND==gemmul8::Backend::INT8){
           gemmul8::gemm<typeC, BACKEND>(
              cublas,
              transa, transb,
              m, n, k,
              (typeC*) alpha, (typeC*) A_d, lda,
              (typeC*) B_d, ldb, (typeC*) zero,
              (typeC*) devCT, ldc,
              num_moduli,
              fastmode, work_rem, workA, workB, enable_skip_scalA, enable_skip_scalB,
              skip_scalA, skip_scalB
           );
           return;
         }
         throw std::runtime_error("No implementation of Ozaki-II wityh FP8 support on H100 (compute capability 90)");
#endif
    }, typeC);
}

void dispatch_gemmul8(
           cublasHandle_t cublas,
           CudaTypeVar typeC, 
           const gemmul8::Backend BACKEND,
           cublasOperation_t transa,
           cublasOperation_t transb,
           size_t m, 
           size_t n, 
           size_t k,
           const void* alpha, 
           const void* A_d, 
           size_t lda,
           const void* B_d, 
           size_t ldb,
           const void* zero, 
           void* devCT, 
           size_t ldc,
           int num_moduli, 
           bool fastmode, 
           void* work_rem, 
           void* workA, 
           void* workB,
           bool enable_skip_scalA, 
           bool enable_skip_scalB, 
           bool skip_scalA, 
           bool skip_scalB)
{
    switch(BACKEND){
       case gemmul8::Backend::INT8:
           dispatch_gemmul8<gemmul8::Backend::INT8>(
              cublas,
              typeC, 
              transa,
              transb,
              m, 
              n, 
              k,
              alpha, 
              A_d, 
              lda,
              B_d, 
              ldb,
              zero, 
              devCT, 
              ldc,
              num_moduli, 
              fastmode, 
              work_rem, 
              workA, 
              workB,
              enable_skip_scalA, 
              enable_skip_scalB, 
              skip_scalA, 
              skip_scalB
          );
          break;
       case gemmul8::Backend::FP8:
           dispatch_gemmul8<gemmul8::Backend::FP8>(
              cublas,
              typeC, 
              transa,
              transb,
              m, 
              n, 
              k,
              alpha, 
              A_d, 
              lda,
              B_d, 
              ldb,
              zero, 
              devCT, 
              ldc,
              num_moduli, 
              fastmode, 
              work_rem, 
              workA, 
              workB,
              enable_skip_scalA, 
              enable_skip_scalB, 
              skip_scalA, 
              skip_scalB
          );
          break;
    }
}

#endif

// ===========================================================================
// Plan creation: the TTGT "optimize" step, computing the transpose plans and
// GEMM configuration from the TAPP tensor descriptors and index strings.
// ===========================================================================
TAPP_error TAPP_create_tensor_product(TAPP_tensor_product* plan,
                                      TAPP_executor exec,
                                      TAPP_element_op op_A,
                                      TAPP_tensor_info A,
                                      const int64_t* idx_A,
                                      TAPP_element_op op_B,
                                      TAPP_tensor_info B,
                                      const int64_t* idx_B,
                                      TAPP_element_op op_C,
                                      TAPP_tensor_info C,
                                      const int64_t* idx_C,
                                      TAPP_element_op op_D,
                                      TAPP_tensor_info D,
                                      const int64_t* idx_D,
                                      TAPP_prectype prec
                                      )
{
    (void)idx_C;
    struct tensor_info* A_info = (struct tensor_info*)A;
    struct tensor_info* B_info = (struct tensor_info*)B;
    struct tensor_info* C_info = (struct tensor_info*)C;
    struct tensor_info* D_info = (struct tensor_info*)D;

    struct product_plan* p = new struct product_plan;
    p->failed = false;
    p->transposeA = p->transposeB = p->transposeC = false;
    p->op_D = op_D;
    p->type_A = A_info->type;
    p->type_B = B_info->type;
    p->type_D = D_info->type;
    p->elements_A = A_info->elements;
    p->elements_B = B_info->elements;
    p->elements_D = D_info->elements;
    if(!translate_prectype(prec, D_info->type, p->compute_type)) return TAPP_ERR_UNSUPPORTED_DATATYPE;

    // The TTGT/cuBLAS-GEMM back-end handles plain (Case 1/2) contractions over
    // dense, positive-stride tensors with identity element-wise ops only. The
    // variable-precision (digit-count) compute types are FP64 emulation, so
    // they require an F64/C64 output.
    if (!is_supported_datatype(A_info->type) ||
        !is_supported_datatype(B_info->type) ||
        !is_supported_datatype(D_info->type) ||
        !tensor_strides_supported(A_info) || !tensor_strides_supported(B_info) ||
        !tensor_strides_supported(C_info) || !tensor_strides_supported(D_info) ||
        (tapp_prec_digits(prec) > 0 &&
         D_info->type != TAPP_F64 && D_info->type != TAPP_C64))
    {
        p->failed = true;
        *plan = (TAPP_tensor_product)p;
        return TAPP_ERR_UNSUPPORTED_DATATYPE;
    }

    // Variable-precision cuBLAS fixed-point FP64 emulation. The requested decimal
    // digits come from the compute type (TAPP_F_*/TAPP_C_*); compute_type was
    // already set to the emulated type by translate_prectype above. The digit
    // count is only consumed when built with EMULATION (CUDA >= 13) - otherwise
    // it stays 0 and the GEMM runs in plain FP64. The digits->mantissa-bits
    // conversion happens at execute.
    // Note: works since emulated types are >= 1000
    p->prec_digits = tapp_prec_digits(prec);

    int rankA = A_info->nmode;
    int rankB = B_info->nmode;
    int rankC = D_info->nmode;  // contraction output is described by D

    // Size with a floor of 1 so .data()/[0] are always valid even for scalars.
    auto sz = [](int r) { return r > 0 ? (size_t)r : (size_t)1; };
    std::vector<int64_t> dimA(sz(rankA)), modeA(sz(rankA));
    std::vector<int64_t> dimB(sz(rankB)), modeB(sz(rankB));
    std::vector<int64_t> dimC(sz(rankC)), modeC(sz(rankC));
    for (int i = 0; i < rankA; i++) { dimA[i] = A_info->extents[i]; modeA[i] = idx_A[i]; }
    for (int i = 0; i < rankB; i++) { dimB[i] = B_info->extents[i]; modeB[i] = idx_B[i]; }
    for (int i = 0; i < rankC; i++) { dimC[i] = D_info->extents[i]; modeC[i] = idx_D[i]; }

    p->m = p->n = p->k = p->lda = p->ldb = p->ldc = 1;

    // --- BASELINE TTGT index scheme ---
    int64_t* bounded = nullptr;
    int n_bounded = 0;
    if (A_info->elements > B_info->elements)
        get_bounded_indices(&bounded, &n_bounded, modeA.data(), rankA, modeC.data(), rankC);
    else
        get_bounded_indices(&bounded, &n_bounded, modeB.data(), rankB, modeC.data(), rankC);

    int n_freeA = rankA - n_bounded;
    int n_freeB = rankB - n_bounded;

    // n_bounded is the contracted-index count of the larger of A/B; the simple
    // TTGT bookkeeping assumes both share it. With repeated index labels (Case
    // 3) the counts disagree and a free-index count can go negative, which would
    // turn the memcpy lengths below into huge sizes. Reject rather than corrupt
    // memory; this case is outside the supported simple-contraction subset.
    if (n_freeA < 0 || n_freeB < 0)
    {
        delete[] bounded;
        p->failed = true;
        *plan = (TAPP_tensor_product)p;
        return TAPP_ERR_UNSUPPORTED_DATATYPE;
    }

    std::vector<int64_t> freeA(n_freeA > 0 ? n_freeA : 1);
    get_free_indices(freeA.data(), n_freeA, modeA.data(), rankA, modeC.data(), rankC);
    std::vector<int64_t> freeB(n_freeB > 0 ? n_freeB : 1);
    get_free_indices(freeB.data(), n_freeB, modeB.data(), rankB, modeC.data(), rankC);

    // modeAT/modeBT always receive exactly rankA/rankB labels. modeCT receives
    // n_freeA+n_freeB, which EXCEEDS rankC when a Hadamard index is shared (it
    // appears in both free sets); size it to the real write length so the copy
    // never overflows. get_permutation reads only the first rankC entries.
    std::vector<int64_t> modeAT(sz(rankA)), modeBT(sz(rankB));
    std::vector<int64_t> modeCT(std::max((size_t)(n_freeA + n_freeB), sz(rankC)));
    std::vector<int> permA(sz(rankA)), permB(sz(rankB)), permC(sz(rankC));
    std::vector<int64_t> dimCT(sz(rankC));

    // Tensor A: keep the leading index in place where possible (col-major).
    bool A_colmajor = A_info->strides[0]==1;
    if(!A_colmajor){
        return TAPP_ERR_NOT_IMPLEMENTED;
    }
    if (std::find(freeA.data(), freeA.data() + n_freeA, modeA[0]) != freeA.data() + n_freeA)
    {
        std::memcpy(modeAT.data(), freeA.data(), n_freeA * sizeof(int64_t));
        std::memcpy(modeAT.data() + n_freeA, bounded, n_bounded * sizeof(int64_t));
        p->transa = CUBLAS_OP_N;
    }
    else
    {
        std::memcpy(modeAT.data(), bounded, n_bounded * sizeof(int64_t));
        std::memcpy(modeAT.data() + n_bounded, freeA.data(), n_freeA * sizeof(int64_t));
        p->transa = CUBLAS_OP_T;
    }
    get_permutation(permA.data(), modeAT.data(), modeA.data(), rankA);
    if (p->transa == CUBLAS_OP_N)
    {
        for (int i = 0; i < n_freeA; i++) p->m *= dimA[permA[i]];
        for (int i = 0; i < n_bounded; i++) p->k *= dimA[permA[i + n_freeA]];
        p->lda = p->m;
    }
    else
    {
        for (int i = 0; i < n_bounded; i++) p->k *= dimA[permA[i]];
        for (int i = 0; i < n_freeA; i++) p->m *= dimA[permA[i + n_bounded]];
        p->lda = p->k;
    }

    // Tensor B.
    bool B_colmajor = B_info->strides[0]==1;
    if(!B_colmajor){
        return TAPP_ERR_NOT_IMPLEMENTED;
    }
    if (std::find(freeB.data(), freeB.data() + n_freeB, modeB[0]) != freeB.data() + n_freeB)
    {
        std::memcpy(modeBT.data(), freeB.data(), n_freeB * sizeof(int64_t));
        std::memcpy(modeBT.data() + n_freeB, bounded, n_bounded * sizeof(int64_t));
        p->transb = CUBLAS_OP_T;
    }
    else
    {
        std::memcpy(modeBT.data(), bounded, n_bounded * sizeof(int64_t));
        std::memcpy(modeBT.data() + n_bounded, freeB.data(), n_freeB * sizeof(int64_t));
        p->transb = CUBLAS_OP_N;
    }
    get_permutation(permB.data(), modeBT.data(), modeB.data(), rankB);
    if (p->transb == CUBLAS_OP_T)
    {
        for (int i = 0; i < n_freeB; i++) p->n *= dimB[permB[i]];
        p->ldb = p->n;
    }
    else
    {
        for (int i = 0; i < n_freeB; i++) p->n *= dimB[permB[i + n_bounded]];
        p->ldb = p->k;
    }

    // Tensor C/D: assume output order [free(A), free(B)].
    bool C_colmajor = C_info->strides[0]==1 or C_info->elements==1;
    if(!C_colmajor){
        return TAPP_ERR_NOT_IMPLEMENTED;
    }
    bool D_colmajor = D_info->strides[0]==1 or D_info->elements==1;
    if(!D_colmajor){
        return TAPP_ERR_NOT_IMPLEMENTED;
    }
    std::memcpy(modeCT.data(), freeA.data(), n_freeA * sizeof(int64_t));
    std::memcpy(modeCT.data() + n_freeA, freeB.data(), n_freeB * sizeof(int64_t));
    get_permutation(permC.data(), modeC.data(), modeCT.data(), rankC);
    for (int i = 0; i < rankC; i++) dimCT[permC[i]] = dimC[i];
    for (int i = 0; i < n_freeA; i++) p->ldc *= dimCT[i];

    delete[] bounded;

    p->transposeA = !std::equal(modeA.begin(), modeA.end(), modeAT.begin());
    p->transposeB = !std::equal(modeB.begin(), modeB.end(), modeBT.begin());
    p->transposeC = !std::equal(modeC.begin(), modeC.end(), modeCT.begin());

    // cuTENSOR plans
    int* permA_p = permA.data();
    int* permB_p = permB.data();
    int* permC_p = permC.data();
    struct executor* exec_struct = (struct executor*) exec;
    TAPP_element_op op_AD = ((op_A==TAPP_CONJUGATE || op_D==TAPP_CONJUGATE) && ~(op_A==TAPP_CONJUGATE && op_D==TAPP_CONJUGATE))? TAPP_CONJUGATE : TAPP_IDENTITY;
    TAPP_element_op op_BD = ((op_B==TAPP_CONJUGATE || op_D==TAPP_CONJUGATE) && ~(op_B==TAPP_CONJUGATE && op_D==TAPP_CONJUGATE))? TAPP_CONJUGATE : TAPP_IDENTITY;
    TAPP_element_op op_CD = ((op_C==TAPP_CONJUGATE || op_D==TAPP_CONJUGATE) && ~(op_C==TAPP_CONJUGATE && op_D==TAPP_CONJUGATE))? TAPP_CONJUGATE : TAPP_IDENTITY;

    p->transposeA = p->transposeA || op_AD!=TAPP_IDENTITY;
    p->transposeB = p->transposeB || op_BD!=TAPP_IDENTITY;
    p->transposeC = p->transposeC || op_CD!=TAPP_IDENTITY;

    // cuTENSOR objects def 
    cutensorStatus_t status;
    cutensorPlanPreference_t pref;
    cutensorTensorDescriptor_t descC;
    cutensorTensorDescriptor_t descCT;
    cutensorOperationDescriptor_t descPermute;
    cutensorOperationDescriptor_t descAccumulation;
    cutensorOperator_t op_cutensor; 
    cutensorOperator_t op_cutensor_CD; 
    cutensorComputeDescriptor_t cutensor_prec;

    std::vector<int32_t> cuidx_C = std::vector<int32_t>(idx_C, idx_C + rankC);

    status = cutensorCreatePlanPreference(exec_struct->cutensor,
                &pref,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // Plan A->op_AD(AT) and B->op_BD(BT) 
    if (p->transposeA){
        status = create_cutensor_permute_plan(exec_struct->cutensor, &p->planA, A_info, modeA.data(), modeAT.data(), op_AD, permA_p);
        if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;
    }
    if (p->transposeB){
        status = create_cutensor_permute_plan(exec_struct->cutensor, &p->planB, B_info, modeB.data(), modeBT.data(), op_BD, permB_p);
        if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;
    }

    // Plan C_T-> C_tilde
    status = cutensorCreateTensorDescriptor(
        exec_struct->cutensor,
        &descC,
        rankC,
        C_info->extents,
        C_info->strides,
        get_cuda_datatype(C_info->type),
        ALIGNMENT_REQUIREMENT
    );
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    if (p->transposeC){
        int64_t stridesCT[rankC];                                                     
        int64_t idx_CT[rankC];                                                     
        for(uint32_t i=0; i<rankC; i++) idx_CT[permC[i]] = idx_C[i];
        stridesCT[0] = 1;
        for(uint32_t i=1; i<rankC; i++) stridesCT[i] = stridesCT[i-1] * dimCT.data()[i-1];

        status = cutensorCreateTensorDescriptor(
            exec_struct->cutensor,
            &descCT,
            rankC,
            dimCT.data(),
            stridesCT,
            get_cuda_datatype(C_info->type),
            ALIGNMENT_REQUIREMENT
        );
        if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

        std::vector<int32_t> cuidx_CT = std::vector<int32_t>(idx_CT, idx_CT + rankC);

        translate_operator(TAPP_IDENTITY, op_cutensor);
        if(!translate_prectype_permute(C_info->type, cutensor_prec)){
            status = CUTENSOR_STATUS_NOT_SUPPORTED;
            goto cleanup;
        }
        status = cutensorCreatePermutation(
            exec_struct->cutensor,
            &descPermute,
            descCT,
            cuidx_CT.data(),
            op_cutensor,
            descC,
            cuidx_C.data(),
            cutensor_prec
        );
        if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;
        status = cutensorCreatePlan(exec_struct->cutensor, &p->planC, descPermute, pref, 0);
        if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

        status = cutensorDestroyOperationDescriptor(descPermute);
        if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;
        status = cutensorDestroyTensorDescriptor(descCT);
        if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;
    }

    // Last plan D = op_D(alpha) C_tilde + op_D(beta) op_CD(C)
    // Note: D, tilde_C and C are assumed to have the same structure
    translate_operator(TAPP_IDENTITY, op_cutensor);
    translate_operator(op_CD, op_cutensor_CD);
    if(!translate_prectype_permute(C_info->type, cutensor_prec)){
        status = CUTENSOR_STATUS_NOT_SUPPORTED;
        goto cleanup;
    }
    status = cutensorCreateElementwiseBinary(
        exec_struct->cutensor,
        &descAccumulation,
        descC,
        cuidx_C.data(),
        op_cutensor,
        descC,
        cuidx_C.data(),
        op_cutensor_CD,
        descC,
        cuidx_C.data(),
        CUTENSOR_OP_ADD,
        cutensor_prec
    );
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    status = cutensorCreatePlan(exec_struct->cutensor, &p->planD, descAccumulation, pref, 0);
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    status = cutensorDestroyOperationDescriptor(descAccumulation);
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    status = cutensorDestroyTensorDescriptor(descC);
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    status = cutensorDestroyPlanPreference(pref);
    if (status!=CUTENSOR_STATUS_SUCCESS) goto cleanup;

    *plan = (TAPP_tensor_product)p;
    return p->failed ? TAPP_ERR_PRODUCT_PLAN_CREATION_FAILED : TAPP_SUCCESS;

cleanup:
    if(pref) cutensorDestroyPlanPreference(pref);
    if(p->planA) cutensorDestroyPlan(p->planA);
    if(p->planB) cutensorDestroyPlan(p->planB);
    if(descC) cutensorDestroyTensorDescriptor(descC);
    if(descCT) cutensorDestroyTensorDescriptor(descCT);
    if(descPermute) cutensorDestroyOperationDescriptor(descPermute);
    if(descAccumulation) cutensorDestroyOperationDescriptor(descAccumulation);
    return tapp_error(status);
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    struct product_plan* p = (struct product_plan*)plan;
    if (p->transposeA) cutensorDestroyPlan(p->planA);
    if (p->transposeB) cutensorDestroyPlan(p->planB);
    if (p->transposeC) cutensorDestroyPlan(p->planC);
    cutensorDestroyPlan(p->planD);
    delete p;
    return TAPP_SUCCESS;
}

// ===========================================================================
// Execution: the TTGT "execute" step. Optionally transposes A/B into matrices,
// runs the GEMM into CT, transposes CT back to D, and adds beta*C.
// ===========================================================================
TAPP_error TAPP_execute_product(TAPP_tensor_product plan,
                                TAPP_executor exec,
                                TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* B,
                                const void* beta,
                                const void* C,
                                      void* D)
{
    struct product_plan* p = (struct product_plan*)plan;
    if (p->failed) return TAPP_ERR_PRODUCT_PLAN_CREATION_FAILED;
    struct executor* exec_struct = (struct executor*) exec;
    cublasHandle_t cublas = exec_struct->cublas;
    cutensorHandle_t cutensor = exec_struct->cutensor;
    cudaStream_t stream = exec_struct->stream;
    TAPP_datatype tD = p->type_D;
    size_t esD = sizeof_datatype(tD);
    size_t bytes_D = p->elements_D * esD;
    cudaError_t cerr;
    cutensorStatus_t cutensor_status;
    cublasStatus_t cublas_status;

    bool beta_nonzero = scalar_is_nonzero(beta, tD);
    if (beta_nonzero && C == nullptr) return TAPP_ERR_C_NULL_BETA;

    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    // Variables used in the function moved due to goto statements
    unsigned char zero[16];
    unsigned char one[16];
    std::complex<double> alpha_buf, beta_buf;
    void *devAT = nullptr, *devBT = nullptr, *devCT = nullptr, *devC_tmp = nullptr;
    void *work;
    TAPP_error error;

#if OZAKI_EMULATION_VERSION==2
    // OS-II variables used in the function moved due to goto statements
    int TAPP_EMULATION_NUM_GEMM = 16;  
    const bool fastmode = false;
    const bool enable_skip_scalA = false;
    const bool enable_skip_scalB = false;
    const bool skip_scalA = false;
    const bool skip_scalB = false;
    const gemmul8::Backend BACKEND = gemmul8::Backend::INT8; // TODO: add support for FP8
    size_t worksizeA, worksizeB;
    size_t worksize;
    size_t offsetA;
    size_t offsetB;
    int8_t *workA    = nullptr;
    int8_t *workB    = nullptr;
    int8_t *work_rem = nullptr;
    TAPP_prectype typeA = p->type_A;
    TAPP_prectype typeB = p->type_B;
    TAPP_prectype typeC = p->type_D;
#endif


    // Create a copy of C if beta!=0 due to cutensorElementwiseBinaryExecute limitation:
    // https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorelementwisebinaryexecute
    void* C_copy = nullptr;
    if (beta_nonzero)
    {
        if (C == D)
        {
            cerr = cudaMallocAsync(&C_copy, p->elements_D * sizeof_datatype(p->type_D), stream);
            if (cerr != cudaSuccess){
                error = tapp_error(cerr);
                goto cleanup;
            }
            cerr = cudaMemcpyAsync(C_copy, C, bytes_D, cudaMemcpyDeviceToDevice, stream);
            if (cerr != cudaSuccess){
                error = tapp_error(cerr);
                goto cleanup;
            }
        } else C_copy = (void*) C;
    } else C_copy = (void*) C; // Should not be needed according to cuTENSOR documentation...

    make_scalar(tD, 0.0, zero);
    make_scalar(tD, 1.0, one);

    if (p->transposeA){
        cerr = cudaMallocAsync(&devAT, p->elements_A * sizeof_datatype(p->type_A), stream);
        if (cerr != cudaSuccess){
            error = tapp_error(cerr);
            goto cleanup;
        }

        cutensor_status = cutensorPermute(
           cutensor,
           p->planA,
           one,
           A,
           devAT,
           stream
        );
        if (cutensor_status != CUTENSOR_STATUS_SUCCESS){
            error = tapp_error(cutensor_status);
            goto cleanup;
        }
    }
    if (p->transposeB){
        cerr = cudaMallocAsync(&devBT, p->elements_B * sizeof_datatype(p->type_B), stream);
        if (cerr != cudaSuccess){
            error = tapp_error(cerr);
            goto cleanup;
        }

        cutensor_status = cutensorPermute(
           cutensor,
           p->planB,
           one,
           B,
           devBT,
           stream
        );
        if (cutensor_status != CUTENSOR_STATUS_SUCCESS){
            error = tapp_error(cutensor_status);
            goto cleanup;
        }
    }

    cerr = cudaMallocAsync(&devCT, bytes_D, stream);
    if (cerr != cudaSuccess){
        error = tapp_error(cerr);
        goto cleanup;
    }
    
    cublasSetStream(cublas, stream);

#if OZAKI_EMULATION_VERSION!=2
#if OZAKI_EMULATION_VERSION==1
        // Configure cuBLAS fixed-point FP64 emulation for the requested number
        // of decimal digits (variable mantissa size). bits = ceil(log2(10)*d).
        if (p->prec_digits > 0)
        {
            if(exec_struct->emulation_strategy == TAPP_EMULATION_STRATEGY_PERFORMANT) {cublasSetEmulationStrategy(cublas, CUBLAS_EMULATION_STRATEGY_PERFORMANT);}
	        else { cublasSetEmulationStrategy(cublas, CUBLAS_EMULATION_STRATEGY_EAGER);}

            if(exec_struct->emulation_mantissa_control == TAPP_EMULATION_MANTISSA_CONTROL_DYNAMIC) {cublasSetFixedPointEmulationMantissaControl(cublas, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);}
	        else{cublasSetFixedPointEmulationMantissaControl(cublas, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);}

            int bits = (int)std::ceil(std::log2(10.0) * p->prec_digits);
            cublasSetFixedPointEmulationMaxMantissaBitCount(cublas, bits);
        }
#endif

    cublas_status = cublasGemmEx_64(
        cublas, p->transa, p->transb, p->m, p->n, p->k,
        one, p->transposeA ? devAT : A, get_cuda_datatype(p->type_A), p->lda,
        p->transposeB ? devBT : B, get_cuda_datatype(p->type_B), p->ldb,
        zero, devCT, get_cuda_datatype(tD), p->ldc, p->compute_type, CUBLAS_GEMM_DEFAULT);
    if(cublas_status!=CUBLAS_STATUS_SUCCESS){
        error = tapp_error(cublas_status);
        goto cleanup;
    }
#else
    // Configure Ozaki-II fixed-point FP64 emulation
    if (p->prec_digits > 0)
    {
       int bits = (int)std::ceil(std::log2(10.0) * p->prec_digits);
       for(const auto& [num_moduli, effective_bits]: OS_II_INT8_table){
          if(effective_bits>=bits){
             TAPP_EMULATION_NUM_GEMM = num_moduli;  
             break;
          }
       }
    }

    if(is_complex(p->type_A) or is_complex(p->type_B) or is_complex(p->type_D)){
      worksize = gemmul8::workSize<true, BACKEND, gemmul8::Func::gemm>(
         p->m, 
         p->n, 
         p->k, 
         TAPP_EMULATION_NUM_GEMM,
         enable_skip_scalA, enable_skip_scalB, &worksizeA, &worksizeB);
    }
    else{
      worksize = gemmul8::workSize<false, BACKEND, gemmul8::Func::gemm>(
         p->m, 
         p->n, 
         p->k, 
         TAPP_EMULATION_NUM_GEMM,
         enable_skip_scalA, enable_skip_scalB, &worksizeA, &worksizeB);
    }

    cerr = cudaMallocAsync(&work, worksize, stream);
    if (cerr != cudaSuccess){
        error = tapp_error(cerr);
        goto cleanup;
    }
   
    offsetA = worksizeA;
    offsetB = worksizeB;
    
    workA    = reinterpret_cast<int8_t *>(work);
    workB    = workA + offsetA;                 
    work_rem = workB + offsetB;
    
    // Dispatch Ozaki-II GEMM
    if(
        not(
            typeA==typeB and 
            typeA==typeC and 
            typeC==typeB
        )
    ) throw std::runtime_error("Ozaki-II does not support tensors of different datatype");
    dispatch_gemmul8(
       cublas,
       get_cpp_datatype(typeC), 
       BACKEND,
       p->transa, p->transb,
       p->m, p->n, p->k,
       one, p->transposeA? devAT: A, p->lda,
       p->transposeB? devBT: B, p->ldb,
       zero, devCT, p->ldc,
       TAPP_EMULATION_NUM_GEMM, fastmode, (void*)work_rem, (void*)workA, (void*)workB,
       enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB
    );

    cudaFreeAsync(work, stream);
#endif

    if (p->transposeA) {
        cerr = cudaFreeAsync(devAT,stream); 
        if (cerr != cudaSuccess){
            error = tapp_error(cerr);
            goto cleanup;
        }
        devAT = nullptr; 
    }
    if (p->transposeB) {
        cerr = cudaFreeAsync(devBT, stream);
        if (cerr != cudaSuccess){
            error = tapp_error(cerr);
            goto cleanup;
        }
        devBT = nullptr; 
    }

    cerr = cudaMallocAsync(&devC_tmp, bytes_D, stream);
    if (cerr != cudaSuccess){
        error = tapp_error(cerr);
        goto cleanup;
    }

    if (p->transposeC){
        cutensor_status = cutensorPermute(
           cutensor,
           p->planC,
           one,
           devCT,
           devC_tmp,
           stream
        );
        if (cutensor_status != CUTENSOR_STATUS_SUCCESS){
            error = tapp_error(cutensor_status);
            goto cleanup;
        }
    }
    else{
        cerr = cudaMemcpyAsync(devC_tmp, devCT, bytes_D, cudaMemcpyDeviceToDevice, stream);
        if (cerr != cudaSuccess){
            error = tapp_error(cerr);
            goto cleanup;
        }
    }
    cerr = cudaFreeAsync(devCT, stream);
    if (cerr != cudaSuccess){
        error = tapp_error(cerr);
        goto cleanup;
    }
    devCT = nullptr;

    // D = beta*C_copy + alpha*C_tmp.
    load_scalar(get_cuda_datatype(p->type_D), alpha, p->op_D==TAPP_CONJUGATE, &alpha_buf);
    load_scalar(get_cuda_datatype(p->type_D), beta,  p->op_D==TAPP_CONJUGATE, &beta_buf);

    cutensor_status =  cutensorElementwiseBinaryExecute(
       cutensor,
        p->planD,
        &alpha_buf,
        devC_tmp,
        &beta_buf,
        C_copy,
        D,
        stream
    );
    if (cutensor_status != CUTENSOR_STATUS_SUCCESS){
        error = tapp_error(cutensor_status);
        goto cleanup;
    }
    cerr = cudaFreeAsync(devC_tmp, stream); 
    if (cerr != cudaSuccess){
        error = tapp_error(cerr);
        goto cleanup;
    }
    devC_tmp = nullptr;

    if (beta_nonzero && C==D){
        cerr = cudaFreeAsync(C_copy, stream);
        if (cerr != cudaSuccess){
            error = tapp_error(cerr);
            goto cleanup;
        }
    }
    C_copy = nullptr;

    error = create_status(exec_struct->stream, exec_struct->device, status);
    if (!TAPP_check_success(error)) goto cleanup;

    return TAPP_SUCCESS;

cleanup:
    if(p->transposeA) if(devAT) cudaFreeAsync(devAT, stream);
    if(p->transposeB) if(devBT) cudaFreeAsync(devBT, stream);
    if(p->transposeC) if(devCT) cudaFreeAsync(devCT, stream);
    if(devC_tmp) cudaFreeAsync(devC_tmp, stream);
    if(work) cudaFreeAsync(work, stream);
    return error;
}
