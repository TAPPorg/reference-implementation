#include "../include/product.h"
#include "../include/status.h"

#include <cuda_runtime.h>

#include <cstring>

// Is the supplied datatype one the TTGT/cuBLAS-GEMM back-end can handle?
static bool is_supported_datatype(TAPP_datatype type)
{
    return type == TAPP_F32 || type == TAPP_F64 ||
           type == TAPP_C32 || type == TAPP_C64;
}

// Is the scalar in `storage` (of the given datatype) non-zero? Used to decide
// whether tensor C must be read.
static bool scalar_is_nonzero(const void* storage, TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
        return *(const float*)storage != 0.0f;
    case TAPP_F64:
        return *(const double*)storage != 0.0;
    case TAPP_C32:
    {
        const float* v = (const float*)storage;
        return v[0] != 0.0f || v[1] != 0.0f;
    }
    case TAPP_C64:
    {
        const double* v = (const double*)storage;
        return v[0] != 0.0 || v[1] != 0.0;
    }
    default:
        return true;
    }
}

TAPP_error TAPP_create_tensor_product(TAPP_tensor_product* plan,
                                      TAPP_handle handle,
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
                                      TAPP_prectype prec)
{
    struct tensor_info* A_info = (struct tensor_info*) A;
    struct tensor_info* B_info = (struct tensor_info*) B;
    struct tensor_info* C_info = (struct tensor_info*) C;
    struct tensor_info* D_info = (struct tensor_info*) D;

    // The TTGT/cuBLAS-GEMM back-end implements plain (Case 1/2) contractions
    // with identity element-wise ops only. Element-wise conjugation and the
    // unsupported (half-precision) datatypes are rejected up front.
    if (op_A != TAPP_IDENTITY || op_B != TAPP_IDENTITY ||
        op_C != TAPP_IDENTITY || op_D != TAPP_IDENTITY)
    {
        return pack_error(0, 16);
    }
    if (!is_supported_datatype(A_info->type) ||
        !is_supported_datatype(B_info->type) ||
        !is_supported_datatype(D_info->type))
    {
        return pack_error(0, 16);
    }

    int nmode_A = A_info->nmode;
    int nmode_B = B_info->nmode;
    int nmode_D = D_info->nmode;

    struct product_plan* plan_struct = new struct product_plan;
    plan_struct->handle = handle;

    // Own the dim/mode arrays referenced by the ContractionInfo. TTGT describes
    // the contraction output by tensor D (idx_D / D extents); C is the additive
    // operand and shares D's structure (TAPP Note 5).
    plan_struct->dimA = new int[nmode_A];
    plan_struct->dimB = new int[nmode_B];
    plan_struct->dimC = new int[nmode_D];
    plan_struct->modeA = new int32_t[nmode_A];
    plan_struct->modeB = new int32_t[nmode_B];
    plan_struct->modeC = new int32_t[nmode_D];
    for (int i = 0; i < nmode_A; i++)
    {
        plan_struct->dimA[i] = (int) A_info->extents[i];
        plan_struct->modeA[i] = (int32_t) idx_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        plan_struct->dimB[i] = (int) B_info->extents[i];
        plan_struct->modeB[i] = (int32_t) idx_B[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        plan_struct->dimC[i] = (int) D_info->extents[i];
        plan_struct->modeC[i] = (int32_t) idx_D[i];
    }

    // alpha/beta are unknown until execute time; point the ContractionInfo at
    // the plan-owned scalar buffers, which TAPP_execute_product fills in.
    plan_struct->info = new ContractionInfo(
        A_info->elements, B_info->elements, D_info->elements,
        plan_struct->dimA, plan_struct->dimB, plan_struct->dimC,
        nmode_A, nmode_B, nmode_D,
        plan_struct->modeA, plan_struct->modeB, plan_struct->modeC,
        translate_datatype(A_info->type),
        translate_datatype(B_info->type),
        translate_datatype(D_info->type),
        plan_struct->alpha_storage, plan_struct->beta_storage,
        translate_prectype(prec, D_info->type));

    plan_struct->ttgt_plan = new TTGTPlan(TransposeBackend::CUTT);
    TTGTOptimizerOptions options;
    try
    {
        plan_struct->ttgt_plan->optimize(options, *plan_struct->info);
    }
    catch (const std::exception& e)
    {
        delete plan_struct->ttgt_plan;
        delete plan_struct->info;
        delete[] plan_struct->dimA;
        delete[] plan_struct->dimB;
        delete[] plan_struct->dimC;
        delete[] plan_struct->modeA;
        delete[] plan_struct->modeB;
        delete[] plan_struct->modeC;
        delete plan_struct;
        return pack_error(0, 16);
    }

    plan_struct->copy_size_A = A_info->copy_size;
    plan_struct->data_offset_A = A_info->data_offset;
    plan_struct->copy_size_B = B_info->copy_size;
    plan_struct->data_offset_B = B_info->data_offset;
    plan_struct->copy_size_C = C_info->copy_size;
    plan_struct->data_offset_C = C_info->data_offset;
    plan_struct->copy_size_D = D_info->copy_size;
    plan_struct->data_offset_D = D_info->data_offset;
    plan_struct->type_D = D_info->type;
    plan_struct->op_D = op_D;

    *plan = (TAPP_tensor_product) plan_struct;
    return 0;
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    struct product_plan* plan_struct = (struct product_plan*) plan;
    delete plan_struct->ttgt_plan;
    delete plan_struct->info;
    delete[] plan_struct->dimA;
    delete[] plan_struct->dimB;
    delete[] plan_struct->dimC;
    delete[] plan_struct->modeA;
    delete[] plan_struct->modeB;
    delete[] plan_struct->modeC;
    delete plan_struct;
    return 0;
}

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
    struct product_plan* plan_struct = (struct product_plan*) plan;
    struct handle* handle_struct = (struct handle*) plan_struct->handle;
    bool use_device_memory = *(bool*)((handle_struct->attributes)[ATTR_KEY_USE_DEVICE_MEMORY]);
    TAPP_datatype type_D = plan_struct->type_D;
    size_t elem_size = sizeof_datatype(type_D);
    cudaError_t cerr;

    // Stage the runtime scalars into the plan-owned buffers the ContractionInfo
    // already points at.
    memcpy(plan_struct->alpha_storage, alpha, elem_size);
    memcpy(plan_struct->beta_storage, beta, elem_size);
    bool beta_nonzero = scalar_is_nonzero(beta, type_D);
    if (beta_nonzero && C == nullptr) return pack_error(0, 12);

    void *A_d, *B_d, *C_d, *D_d;       // logical element-0 pointers
    void *A_base = nullptr, *B_base = nullptr, *C_base = nullptr, *D_base = nullptr;

    if (use_device_memory)
    {
        A_d = (void*)A;
        B_d = (void*)B;
        C_d = (void*)C;
        D_d = (void*)D;
    }
    else
    {
        cerr = cudaMalloc(&A_base, plan_struct->copy_size_A);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMalloc(&B_base, plan_struct->copy_size_B);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMalloc(&D_base, plan_struct->copy_size_D);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMemcpy(A_base, (void*)((intptr_t)A + plan_struct->data_offset_A), plan_struct->copy_size_A, cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMemcpy(B_base, (void*)((intptr_t)B + plan_struct->data_offset_B), plan_struct->copy_size_B, cudaMemcpyHostToDevice);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        A_d = (void*)((intptr_t)A_base + plan_struct->data_offset_A);
        B_d = (void*)((intptr_t)B_base + plan_struct->data_offset_B);
        D_d = (void*)((intptr_t)D_base + plan_struct->data_offset_D);
        C_d = D_d; // C is folded into D below when needed
    }

    // TTGT computes D = alpha*A*B + beta*D in place, so when beta != 0 the
    // contents of C must first be staged into the D buffer.
    if (beta_nonzero)
    {
        if (use_device_memory)
        {
            if (C_d != D_d)
            {
                cerr = cudaMemcpy(D_d, C_d, plan_struct->copy_size_D, cudaMemcpyDeviceToDevice);
                if (cerr != cudaSuccess) return pack_error(0, cerr);
            }
        }
        else
        {
            cerr = cudaMemcpy(D_base, (void*)((intptr_t)C + plan_struct->data_offset_C), plan_struct->copy_size_D, cudaMemcpyHostToDevice);
            if (cerr != cudaSuccess) return pack_error(0, cerr);
        }
    }

    try
    {
        plan_struct->ttgt_plan->execute(A_d, B_d, D_d, *plan_struct->info);
    }
    catch (const std::exception& e)
    {
        return pack_error(0, 16);
    }

    cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) return pack_error(0, cerr);

    if (!use_device_memory)
    {
        cerr = cudaMemcpy((void*)((intptr_t)D + plan_struct->data_offset_D), D_base, plan_struct->copy_size_D, cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        if (A_base) cudaFree(A_base);
        if (B_base) cudaFree(B_base);
        if (D_base) cudaFree(D_base);
    }

    TAPP_error status_err = create_status(*(cudaStream_t*)exec, status);
    if (!TAPP_check_success(status_err)) return status_err;

    cerr = cudaStreamSynchronize(*(cudaStream_t*)exec);
    if (cerr != cudaSuccess) return pack_error(0, cerr);

    return 0;
}
