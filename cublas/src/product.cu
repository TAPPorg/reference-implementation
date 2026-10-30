#include "internal.h"
#include "tapp/executor.h"

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cutt.h>
#include <cuda_runtime.h>
#if OZAKI_EMULATION_VERSION==2
#include <gemmul8.hpp>
#endif

#if OZAKI_EMULATION_VERSION==2
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

// ===========================================================================
// CUDA / cuBLAS / cuTT error checks. Unlike the upstream my-ttgt macros these throw
// instead of exit()-ing, so the binding can report a TAPP error and keep the
// host process alive.
// ===========================================================================

// Carries the TAPP error code of the failed call to the catch in TAPP_execute_product
struct tapp_exception : std::runtime_error
{
    TAPP_error code;
    tapp_exception(TAPP_error code, const std::string& what) : std::runtime_error(what), code(code) {}
};

static inline void cuda_check(cudaError_t code)
{
    if (code != cudaSuccess)
        throw tapp_exception(tapp_error(code), std::string("CUDA error: ") + cudaGetErrorString(code));
}

static inline void cublas_check(cublasStatus_t stat)
{
    if (stat != CUBLAS_STATUS_SUCCESS)
        throw tapp_exception(tapp_error(stat), std::string("cuBLAS error: ") + cublasGetStatusString(stat));
}

static inline void cutt_check(cuttResult err)
{
    if (err != CUTT_SUCCESS)
        throw tapp_exception(tapp_error(err), "cuTT error " + std::to_string((int)err));
}

// ===========================================================================
// Index/permutation helpers implementing the TTGT index bookkeeping.
// ===========================================================================

// Contracted ("bounded") labels of `mode`: those not present in `modeC`.
static void get_bounded_indices(int** bounded, int* n_bounded,
                                const int* mode, int rank,
                                const int* modeC, int rankC)
{
    std::vector<int> tmp(rank);
    *n_bounded = 0;
    for (int i = 0; i < rank; i++)
    {
        int idx = mode[i];
        if (std::find(modeC, modeC + rankC, idx) == modeC + rankC)
            tmp[(*n_bounded)++] = idx;
    }
    *bounded = new int[*n_bounded > 0 ? *n_bounded : 1];
    for (int i = 0; i < *n_bounded; i++)
        (*bounded)[i] = tmp[i];
}

// Free labels of `mode`: those also present in `modeC` (first n_free of them).
static void get_free_indices(int* free_idx, int n_free,
                             const int* mode, int rank,
                             const int* modeC, int rankC)
{
    int index = 0;
    for (int i = 0; i < rank && index < n_free; i++)
    {
        int idx = mode[i];
        if (std::find(modeC, modeC + rankC, idx) != modeC + rankC)
            free_idx[index++] = idx;
    }
}

// permutation[i] = position, in `mode`, of the label sitting at modeT[i].
static void get_permutation(int* permutation, const int* mode,
                            const int* modeT, int rank)
{
    std::unordered_map<int, int> pos;
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
static bool cutt_plan_squeezed(cudaStream_t stream, cuttHandle* plan, int rank, const int* dim,
                               const int* permutation, size_t sizeofType,
                               const char* which)
{
    std::vector<int> old_to_new(rank, -1);
    std::vector<int> sdim;
    int new_rank = 0;
    for (int i = 0; i < rank; i++)
        if (dim[i] != 1)
        {
            old_to_new[i] = new_rank++;
            sdim.push_back(dim[i]);
        }

    std::vector<int> sperm;
    for (int i = 0; i < rank; i++)
    {
        int axis = permutation[i];
        if (dim[axis] != 1)
            sperm.push_back(old_to_new[axis]);
    }

    bool identity = true;
    for (int i = 0; i < new_rank; i++)
        if (sperm[i] != i) { identity = false; break; }
    if (new_rank < 2 || identity)
        return false;

    cuttResult err = cuttPlan(plan, new_rank, sdim.data(), sperm.data(), sizeofType, stream);
    if (err != CUTT_SUCCESS)
    {
        fprintf(stderr,
                "TTGT: cuttPlan failed for tensor %s (err=%d): rank=%d "
                "elemsize=%zu dim=[",
                which, err, new_rank, sizeofType);
        for (int i = 0; i < new_rank; i++)
            fprintf(stderr, "%d%s", sdim[i], i + 1 < new_rank ? "," : "");
        fprintf(stderr, "] perm=[");
        for (int i = 0; i < new_rank; i++)
            fprintf(stderr, "%d%s", sperm[i], i + 1 < new_rank ? "," : "");
        fprintf(stderr, "]\n");
        throw std::runtime_error("cuTT plan creation failed");
    }
    return true;
}

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
#if OZAKI_II_COMPUTE_CAPABILITY!=90
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
    p->copy_size_A = A_info->copy_size;  p->data_offset_A = A_info->data_offset;
    p->copy_size_B = B_info->copy_size;  p->data_offset_B = B_info->data_offset;
    p->copy_size_C = C_info->copy_size;  p->data_offset_C = C_info->data_offset;
    p->copy_size_D = D_info->copy_size;  p->data_offset_D = D_info->data_offset;
    p->compute_type = translate_prectype(prec, D_info->type);

    // The TTGT/cuBLAS-GEMM back-end handles plain (Case 1/2) contractions over
    // dense, positive-stride tensors with identity element-wise ops only. The
    // variable-precision (digit-count) compute types are FP64 emulation, so
    // they require an F64/C64 output.
    if (op_A != TAPP_IDENTITY || op_B != TAPP_IDENTITY ||
        op_C != TAPP_IDENTITY || op_D != TAPP_IDENTITY ||
        !is_supported_datatype(A_info->type) ||
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
    p->prec_digits = 0;
#if OZAKI_EMULATION_VERSION==1
    p->prec_digits = tapp_prec_digits(prec);
#endif

    int rankA = A_info->nmode;
    int rankB = B_info->nmode;
    int rankC = D_info->nmode;  // contraction output is described by D

    // Size with a floor of 1 so .data()/[0] are always valid even for scalars.
    auto sz = [](int r) { return r > 0 ? (size_t)r : (size_t)1; };
    std::vector<int> dimA(sz(rankA)), modeA(sz(rankA));
    std::vector<int> dimB(sz(rankB)), modeB(sz(rankB));
    std::vector<int> dimC(sz(rankC)), modeC(sz(rankC));
    for (int i = 0; i < rankA; i++) { dimA[i] = (int)A_info->extents[i]; modeA[i] = (int)idx_A[i]; }
    for (int i = 0; i < rankB; i++) { dimB[i] = (int)B_info->extents[i]; modeB[i] = (int)idx_B[i]; }
    for (int i = 0; i < rankC; i++) { dimC[i] = (int)D_info->extents[i]; modeC[i] = (int)idx_D[i]; }

    p->m = p->n = p->k = p->lda = p->ldb = p->ldc = 1;

    // --- BASELINE TTGT index scheme ---
    int* bounded = nullptr;
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

    std::vector<int> freeA(n_freeA > 0 ? n_freeA : 1);
    get_free_indices(freeA.data(), n_freeA, modeA.data(), rankA, modeC.data(), rankC);
    std::vector<int> freeB(n_freeB > 0 ? n_freeB : 1);
    get_free_indices(freeB.data(), n_freeB, modeB.data(), rankB, modeC.data(), rankC);

    // modeAT/modeBT always receive exactly rankA/rankB labels. modeCT receives
    // n_freeA+n_freeB, which EXCEEDS rankC when a Hadamard index is shared (it
    // appears in both free sets); size it to the real write length so the copy
    // never overflows. get_permutation reads only the first rankC entries.
    std::vector<int> modeAT(sz(rankA)), modeBT(sz(rankB));
    std::vector<int> modeCT(std::max((size_t)(n_freeA + n_freeB), sz(rankC)));
    std::vector<int> permA(sz(rankA)), permB(sz(rankB)), permC(sz(rankC));
    std::vector<int> dimCT(sz(rankC));

    // Tensor A: keep the leading index in place where possible (col-major).
    if (std::find(freeA.data(), freeA.data() + n_freeA, modeA[0]) != freeA.data() + n_freeA)
    {
        std::memcpy(modeAT.data(), freeA.data(), n_freeA * sizeof(int));
        std::memcpy(modeAT.data() + n_freeA, bounded, n_bounded * sizeof(int));
        p->transa = CUBLAS_OP_N;
    }
    else
    {
        std::memcpy(modeAT.data(), bounded, n_bounded * sizeof(int));
        std::memcpy(modeAT.data() + n_bounded, freeA.data(), n_freeA * sizeof(int));
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
    if (std::find(freeB.data(), freeB.data() + n_freeB, modeB[0]) != freeB.data() + n_freeB)
    {
        std::memcpy(modeBT.data(), freeB.data(), n_freeB * sizeof(int));
        std::memcpy(modeBT.data() + n_freeB, bounded, n_bounded * sizeof(int));
        p->transb = CUBLAS_OP_T;
    }
    else
    {
        std::memcpy(modeBT.data(), bounded, n_bounded * sizeof(int));
        std::memcpy(modeBT.data() + n_bounded, freeB.data(), n_freeB * sizeof(int));
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
    std::memcpy(modeCT.data(), freeA.data(), n_freeA * sizeof(int));
    std::memcpy(modeCT.data() + n_freeA, freeB.data(), n_freeB * sizeof(int));
    get_permutation(permC.data(), modeC.data(), modeCT.data(), rankC);
    for (int i = 0; i < rankC; i++) dimCT[permC[i]] = dimC[i];
    for (int i = 0; i < n_freeA; i++) p->ldc *= dimCT[i];

    delete[] bounded;

    p->transposeA = !std::equal(modeA.begin(), modeA.end(), modeAT.begin());
    p->transposeB = !std::equal(modeB.begin(), modeB.end(), modeBT.begin());
    p->transposeC = !std::equal(modeC.begin(), modeC.end(), modeCT.begin());

    // cuTT plan inputs (A and B transpose from their own layout, C from CT).
    int rA = rankA, rB = rankB, rC = rankC;
    int* dimA_p = dimA.data();   int* permA_p = permA.data();
    int* dimB_p = dimB.data();   int* permB_p = permB.data();
    int* dimCT_p = dimCT.data(); int* permC_p = permC.data();
    size_t es_A = sizeof_datatype(p->type_A);
    size_t es_B = sizeof_datatype(p->type_B);
    size_t es_C = sizeof_datatype(p->type_D);

    // complex64 needs an extra leading dimension of 2 (cuTT element-size limit).
    int *cA = nullptr, *cpA = nullptr, *cB = nullptr, *cpB = nullptr, *cC = nullptr, *cpC = nullptr;
    if (p->type_A == TAPP_C64) { rA++; cA = prepend(dimA.data(), rankA, 2); dimA_p = cA; cpA = shift_permutation(permA.data(), rankA); permA_p = cpA; es_A /= 2; }
    if (p->type_B == TAPP_C64) { rB++; cB = prepend(dimB.data(), rankB, 2); dimB_p = cB; cpB = shift_permutation(permB.data(), rankB); permB_p = cpB; es_B /= 2; }
    if (p->type_D == TAPP_C64) { rC++; cC = prepend(dimCT.data(), rankC, 2); dimCT_p = cC; cpC = shift_permutation(permC.data(), rankC); permC_p = cpC; es_C /= 2; }

    struct executor* exec_struct = reinterpret_cast<struct executor*>(exec);
    p->stream = exec_struct->stream;
    try
    {
        if (p->transposeA) p->transposeA = cutt_plan_squeezed(exec_struct->stream, &p->planA, rA, dimA_p, permA_p, es_A, "A");
        if (p->transposeB) p->transposeB = cutt_plan_squeezed(exec_struct->stream, &p->planB, rB, dimB_p, permB_p, es_B, "B");
        if (p->transposeC) p->transposeC = cutt_plan_squeezed(exec_struct->stream, &p->planC, rC, dimCT_p, permC_p, es_C, "C");
    }
    catch (const std::exception&)
    {
        p->failed = true;
        p->transposeA = p->transposeB = p->transposeC = false;
    }

    delete[] cA; delete[] cpA; delete[] cB; delete[] cpB; delete[] cC; delete[] cpC;

    *plan = (TAPP_tensor_product)p;
    return p->failed ? TAPP_ERR_UNSUPPORTED_DATATYPE : TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    struct product_plan* p = (struct product_plan*)plan;
    if (p->transposeA) cuttDestroy(p->planA);
    if (p->transposeB) cuttDestroy(p->planB);
    if (p->transposeC) cuttDestroy(p->planC);
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
    struct executor* exec_struct = (struct executor*) exec;

    if (p->failed) return TAPP_ERR_UNSUPPORTED_DATATYPE;
    if (p->stream != exec_struct->stream) return TAPP_ERR_PLAN_WRONG_EXECUTOR;
    cublasHandle_t cublas = exec_struct->cublas;
    TAPP_datatype tD = p->type_D;
    size_t esD = sizeof_datatype(tD);
    size_t bytes_D = p->elements_D * esD;
    cudaError_t cerr;

    bool beta_nonzero = scalar_is_nonzero(beta, tD);
    if (beta_nonzero && C == nullptr) return TAPP_ERR_C_NULL_BETA;

    // TTGT works in place on the output buffer, so when beta != 0 stage C into D.
    if (beta_nonzero)
    {
        if (C != D)
        {
            cerr = cudaMemcpyAsync(D, (void*)C, bytes_D, cudaMemcpyDeviceToDevice, exec_struct->stream);
            if (cerr != cudaSuccess) return tapp_error(cerr);
        }
    }

    void *devAT = nullptr, *devBT = nullptr, *devCT = nullptr, *devC_tmp = nullptr;
    TAPP_error result = TAPP_SUCCESS;
    try
    {
        if (p->transposeA) cuda_check(cudaMallocAsync(&devAT, p->elements_A * sizeof_datatype(p->type_A), exec_struct->stream));
        if (p->transposeB) cuda_check(cudaMallocAsync(&devBT, p->elements_B * sizeof_datatype(p->type_B), exec_struct->stream));
        if (p->transposeA) cutt_check(cuttExecute(p->planA, (void*)A, devAT));
        if (p->transposeB) cutt_check(cuttExecute(p->planB, (void*)B, devBT));

        cuda_check(cudaMallocAsync(&devCT, bytes_D, exec_struct->stream));

        unsigned char zero[16];
        make_scalar(tD, 0.0, zero);
        cublasStatus_t stat;

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

        stat = cublasGemmEx(
            cublas, p->transa, p->transb, p->m, p->n, p->k,
            alpha, p->transposeA ? devAT : A_d, get_cuda_datatype(p->type_A), p->lda,
            p->transposeB ? devBT : B_d, get_cuda_datatype(p->type_B), p->ldb,
            zero, devCT, get_cuda_datatype(tD), p->ldc, p->compute_type, CUBLAS_GEMM_DEFAULT);
        cublas_check(stat);
#else
        // Configure Ozaki-II fixed-point FP64 emulation
        int TAPP_EMULATION_NUM_GEMM = exec_struct->num_gemm;
        const bool fastmode = false;
        const bool enable_skip_scalA = false;
        const bool enable_skip_scalB = false;
        const bool skip_scalA = false;
        const bool skip_scalB = false;
        // TODO: add support for FP8
        const gemmul8::Backend BACKEND = gemmul8::Backend::INT8;

        TAPP_prectype typeA = p->type_A;
        TAPP_prectype typeB = p->type_B;
        TAPP_prectype typeC = p->type_D;

        // Checked before the workspace is allocated, which the catch below does not free
        if(
            not(
                typeA==typeB and
                typeA==typeC and
                typeC==typeB
            )
        ) throw std::runtime_error("Ozaki-II does not support tensors of different datatype");

        // Estimate size and create workspace
        size_t worksizeA, worksizeB;
        size_t worksize;
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

        void *work;
        cuda_check(cudaMallocAsync(&work, worksize, exec_struct->stream));
   
        const size_t offsetA = worksizeA;
        const size_t offsetB = worksizeB;
        
        int8_t *workA    = reinterpret_cast<int8_t *>(work);
        int8_t *workB    = workA + offsetA;                 
        int8_t *work_rem = workB + offsetB;
        dispatch_gemmul8(
           cublas,
           get_cpp_datatype(typeC), 
           BACKEND,
           p->transa, p->transb,
           p->m, p->n, p->k,
           alpha, p->transposeA? devAT: A, p->lda,
           p->transposeB? devBT: B, p->ldb,
           zero, devCT, p->ldc,
           TAPP_EMULATION_NUM_GEMM, fastmode, (void*)work_rem, (void*)workA, (void*)workB,
           enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB
        );
        // GEMMul8 does not check its kernel launches, so pick up a launch error here. Checked after freeing the
        // workspace, which the catch below does not know about
        cudaError_t launch_err = cudaGetLastError();

        cudaFreeAsync(work, exec_struct->stream);
        cuda_check(launch_err);
#endif

        if (p->transposeA) { cuda_check(cudaFreeAsync(devAT, exec_struct->stream)); devAT = nullptr; }
        if (p->transposeB) { cuda_check(cudaFreeAsync(devBT, exec_struct->stream)); devBT = nullptr; }

        cuda_check(cudaMallocAsync(&devC_tmp, bytes_D, exec_struct->stream));
        if (p->transposeC) cutt_check(cuttExecute(p->planC, devCT, devC_tmp));
        else cuda_check(cudaMemcpyAsync(devC_tmp, devCT, bytes_D, cudaMemcpyDeviceToDevice, exec_struct->stream));
        cuda_check(cudaFreeAsync(devCT, exec_struct->stream)); devCT = nullptr;

        // D = beta*D + C_tmp (D already holds staged C when beta != 0).
        if (beta_nonzero)
        {
            unsigned char one[16];
            make_scalar(tD, 1.0, one);
            switch (tD)
            {
            case TAPP_F64:
                stat = cublasDgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N, p->m, p->n,
                    (const double*)beta, (const double*)D, p->m,
                    (const double*)one, (const double*)devC_tmp, p->m,
                    (double*)D, p->m);
                break;
            case TAPP_F32:
                stat = cublasSgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N, p->m, p->n,
                    (const float*)beta, (const float*)D, p->m,
                    (const float*)one, (const float*)devC_tmp, p->m,
                    (float*)D, p->m);
                break;
            case TAPP_C64:
                stat = cublasZgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N, p->m, p->n,
                    (const cuDoubleComplex*)beta, (const cuDoubleComplex*)D, p->m,
                    (const cuDoubleComplex*)one, (const cuDoubleComplex*)devC_tmp, p->m,
                    (cuDoubleComplex*)D, p->m);
                break;
            case TAPP_C32:
                stat = cublasCgeam(cublas, CUBLAS_OP_N, CUBLAS_OP_N, p->m, p->n,
                    (const cuComplex*)beta, (const cuComplex*)D, p->m,
                    (const cuComplex*)one, (const cuComplex*)devC_tmp, p->m,
                    (cuComplex*)D, p->m);
                break;
            }
            cublas_check(stat);
        }
        else
        {
            cuda_check(cudaMemcpyAsync(D, devC_tmp, bytes_D, cudaMemcpyDeviceToDevice, exec_struct->stream));
        }
        cuda_check(cudaFreeAsync(devC_tmp, exec_struct->stream)); devC_tmp = nullptr;
    }
    catch (const tapp_exception& e)
    {
        result = e.code;
    }
    catch (const std::exception&)
    {
        // Thrown for contractions the cuBLAS/TTGT back-end does not support, e.g. mixed datatypes with Ozaki-II
        result = TAPP_ERR_UNSUPPORTED_DATATYPE;
    }

    if (result != TAPP_SUCCESS)
    {
        if (devAT) cudaFreeAsync(devAT, exec_struct->stream);
        if (devBT) cudaFreeAsync(devBT, exec_struct->stream);
        if (devCT) cudaFreeAsync(devCT, exec_struct->stream);
        if (devC_tmp) cudaFreeAsync(devC_tmp, exec_struct->stream);
        return result;
    }

    TAPP_error status_err = create_status(exec_struct->stream, status);
    if (status_err != TAPP_SUCCESS) return status_err;

    return TAPP_SUCCESS;
}
