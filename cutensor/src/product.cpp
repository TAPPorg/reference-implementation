#include "tapp/product.h"
#include "internal.h"

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
                                      TAPP_prectype prec)
{
    // Validate the arguments before creating anything, so these errors need no cleanup
    if (!plan || !exec || !A || !B || !C || !D) return TAPP_ERR_NULL_POINTER;

    const struct tensor_info* info_A = (struct tensor_info*)A;
    const struct tensor_info* info_B = (struct tensor_info*)B;
    const struct tensor_info* info_C = (struct tensor_info*)C;
    const struct tensor_info* info_D = (struct tensor_info*)D;
    if ((info_A->nmode > 0 && !idx_A) || (info_B->nmode > 0 && !idx_B) || (info_C->nmode > 0 && !idx_C) || (info_D->nmode > 0 && !idx_D))
    {
        return TAPP_ERR_NULL_POINTER;
    }
    if (info_A->type != info_D->type || info_B->type != info_D->type || info_C->type != info_D->type) return TAPP_ERR_INVALID_DATATYPE;

    // cuTENSOR computes D in the layout of C, so they must have the same structure
    if (!same_structure((struct tensor_info*)C, idx_C, (struct tensor_info*)D, idx_D)) return TAPP_ERR_C_D_MISMATCH;

    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    std::vector<int32_t> cuidx_A = std::vector<int32_t>(idx_A, idx_A + info_A->nmode);
    std::vector<int32_t> cuidx_B = std::vector<int32_t>(idx_B, idx_B + info_B->nmode);
    std::vector<int32_t> cuidx_D = std::vector<int32_t>(idx_D, idx_D + info_D->nmode);

    const bool flip = (op_D == TAPP_CONJUGATE);
    const TAPP_element_op op_A_eff = ((op_A == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY;
    const TAPP_element_op op_B_eff = ((op_B == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY;
    const TAPP_element_op op_C_eff = ((op_C == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY;

    cudaError_t cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    // Resources are null-initialized so the fail: path only releases what was created.
    cutensorTensorDescriptor_t descA = nullptr;
    cutensorTensorDescriptor_t descB = nullptr;
    cutensorTensorDescriptor_t descD = nullptr;
    cutensorOperationDescriptor_t desc = nullptr;
    cutensorPlanPreference_t plan_pref = nullptr;
    cutensorPlan_t cuplan = nullptr;
    cutensorStatus_t err;
    TAPP_error result = TAPP_SUCCESS; // Set for errors that are not from cuTENSOR
    cutensorDataType_t scalarType;
    uint64_t workspace_size_estimate = 0;

    err = cutensorCreateTensorDescriptor(libhandle,
                &descA,
                info_A->nmode,
                info_A->extents,
                info_A->strides,
                translate_datatype(info_A->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    err = cutensorCreateTensorDescriptor(libhandle,
                &descB,
                info_B->nmode,
                info_B->extents,
                info_B->strides,
                translate_datatype(info_B->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // C has the layout of D, so C needs no descriptor of its own
    err = cutensorCreateTensorDescriptor(libhandle,
                &descD,
                info_D->nmode,
                info_D->extents,
                info_D->strides,
                translate_datatype(info_D->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    err = cutensorCreateContraction(libhandle,
                &desc,
                descA, cuidx_A.data(), translate_operator(op_A_eff),
                descB, cuidx_B.data(), translate_operator(op_B_eff),
                descD, cuidx_D.data(), translate_operator(op_C_eff),
                descD, cuidx_D.data(),
                translate_prectype(prec, ((struct tensor_info*)D)->type));
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    err = cutensorOperationDescriptorGetAttribute(libhandle,
                desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType));
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // execute passes alpha and beta in the datatype of D, so cuTENSOR must expect that type
    if (scalarType != translate_datatype(info_D->type))
    {
        result = TAPP_ERR_UNSUPPORTED_PRECISION;
        goto cleanup;
    }

    err = cutensorCreatePlanPreference(
                libhandle,
                &plan_pref,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (err != CUTENSOR_STATUS_SUCCESS) { plan_pref = nullptr; goto cleanup; }

    err = cutensorEstimateWorkspaceSize(libhandle,
                desc,
                plan_pref,
                CUTENSOR_WORKSPACE_DEFAULT,
                &workspace_size_estimate);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    err = cutensorCreatePlan(libhandle,
                &cuplan,
                desc,
                plan_pref,
                workspace_size_estimate);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    {
        struct product_plan* plan_struct = new struct product_plan;
        plan_struct->plan = cuplan;
        plan_struct->type_D = info_D->type;
        plan_struct->conj_scalars = flip;
        *plan = (TAPP_tensor_product) plan_struct;
        cuplan = nullptr; // Owned by plan_struct now, so cleanup leaves it
    }

cleanup:
    if (cuplan) cutensorDestroyPlan(cuplan);
    if (desc) cutensorDestroyOperationDescriptor(desc);
    if (plan_pref) cutensorDestroyPlanPreference(plan_pref);
    if (descA) cutensorDestroyTensorDescriptor(descA);
    if (descB) cutensorDestroyTensorDescriptor(descB);
    if (descD) cutensorDestroyTensorDescriptor(descD);
    return result != TAPP_SUCCESS ? result : tapp_error(err);
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    struct product_plan* plan_struct = (struct product_plan*) plan;
    cutensorStatus_t err;
    err = cutensorDestroyPlan(plan_struct->plan);

    delete plan_struct;
    return tapp_error(err);
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
    if (!plan || !exec || !alpha || !beta || !A || !B || !D) return TAPP_ERR_NULL_POINTER;

    struct product_plan* plan_struct = (struct product_plan*) plan;

    // C is only read when beta is not zero
    if (!C && !is_zero_scalar(plan_struct->type_D, beta)) return TAPP_ERR_NULL_POINTER;

    if (!is_aligned(A) || !is_aligned(B) || !is_aligned(C) || !is_aligned(D)) return TAPP_ERR_MISALIGNED;

    // op_D is folded into op_A, op_B and op_C, so a conjugated D also conjugates the scalars
    std::complex<double> alpha_buf, beta_buf; // Room for a scalar of any type
    load_scalar(plan_struct->type_D, alpha, plan_struct->conj_scalars, &alpha_buf);
    load_scalar(plan_struct->type_D, beta, plan_struct->conj_scalars, &beta_buf);

    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    const cudaStream_t stream = exec_struct->stream;
    cudaError_t cerr;
    cutensorStatus_t err;

    cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    void *workspace = nullptr;
    uint64_t workspace_size = 0;

    err = cutensorPlanGetAttribute(libhandle,
                plan_struct->plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &workspace_size,
                sizeof(workspace_size));
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    workspace_size = std::max(workspace_size, uint64_t(128 * 1024 * 1024)); // 128 MiB recommended minimum size https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcontract
    cerr = cudaMallocAsync(&workspace, workspace_size, stream);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    err = cutensorContract(libhandle,
                plan_struct->plan,
                &alpha_buf, A, B,
                &beta_buf,  C ? C : D, D, // cuTENSOR refuses NULL even when beta is zero and C is not read, D has the same layout
                workspace, workspace_size, stream);
    cerr = cudaFreeAsync(workspace, stream);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    return create_status(stream, exec_struct->device, status);
}
