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
    if ((info_A->extents.size() > 0 && !idx_A) || (info_B->extents.size() > 0 && !idx_B) || (info_C->extents.size() > 0 && !idx_C) || (info_D->extents.size() > 0 && !idx_D))
    {
        return TAPP_ERR_NULL_POINTER;
    }
    if (info_A->type != info_D->type || info_B->type != info_D->type || info_C->type != info_D->type) return TAPP_ERR_INVALID_DATATYPE;
    if (!is_valid_op(op_A) || !is_valid_op(op_B) || !is_valid_op(op_C) || !is_valid_op(op_D))
    {
        return TAPP_ERR_UNSUPPORTED_ELEMENT_OP;
    }

    // cuTENSOR computes D in the layout of C, so they must have the same structure
    if (!same_structure(info_C, idx_C, info_D, idx_D)) return TAPP_ERR_C_D_MISMATCH;

    cutensorComputeDescriptor_t cuprec;
    if (!translate_prectype(prec, info_D->type, cuprec)) return TAPP_ERR_UNSUPPORTED_PRECISION;

    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    std::vector<int32_t> cuidx_A = std::vector<int32_t>(idx_A, idx_A + info_A->extents.size());
    std::vector<int32_t> cuidx_B = std::vector<int32_t>(idx_B, idx_B + info_B->extents.size());
    std::vector<int32_t> cuidx_D = std::vector<int32_t>(idx_D, idx_D + info_D->extents.size());

    const bool flip = (op_D == TAPP_CONJUGATE);
    cutensorOperator_t cuop_A;
    cutensorOperator_t cuop_B;
    cutensorOperator_t cuop_C;
    // Cannot fail: the ops were validated above and folding gives IDENTITY or CONJUGATE
    translate_operator(((op_A == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY, cuop_A);
    translate_operator(((op_B == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY, cuop_B);
    translate_operator(((op_C == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY, cuop_C);

    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    // The cuTENSOR objects below destroy themselves when the function returns, also on errors
    cutensorStatus_t err;

    tensor_descriptor descA;
    err = create_descriptor(libhandle, info_A, descA);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    tensor_descriptor descB;
    err = create_descriptor(libhandle, info_B, descB);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    // C has the layout of D, so C needs no descriptor of its own
    tensor_descriptor descD;
    err = create_descriptor(libhandle, info_D, descD);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    operation_descriptor desc;
    err = cutensorCreateContraction(libhandle,
                &desc.handle,
                descA.handle, cuidx_A.data(), cuop_A,
                descB.handle, cuidx_B.data(), cuop_B,
                descD.handle, cuidx_D.data(), cuop_C,
                descD.handle, cuidx_D.data(),
                cuprec);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    cutensorDataType_t scalar_type;
    err = cutensorOperationDescriptorGetAttribute(libhandle,
                desc.handle,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalar_type,
                sizeof(scalar_type));
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    // execute passes alpha and beta in the datatype of D, so cuTENSOR must expect that type
    if (scalar_type != info_D->type) return TAPP_ERR_UNSUPPORTED_PRECISION;

    plan_preference plan_pref;
    err = cutensorCreatePlanPreference(libhandle,
                &plan_pref.handle,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    uint64_t workspace_size_estimate = 0;
    err = cutensorEstimateWorkspaceSize(libhandle,
                desc.handle,
                plan_pref.handle,
                CUTENSOR_WORKSPACE_DEFAULT,
                &workspace_size_estimate);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    // cuTENSOR recommends some workspace (e.g. 128 MiB) for contractions; the plan picks its algorithm within this limit
    // and execute only provides what the chosen algorithm requires
    // https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcontract
    const uint64_t workspace_limit = std::max(workspace_size_estimate, uint64_t(128) << 20);

    plan_owner cuplan;
    err = cutensorCreatePlan(libhandle, &cuplan.handle, desc.handle, plan_pref.handle, workspace_limit);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    uint64_t workspace_size = 0;
    err = cutensorPlanGetAttribute(libhandle,
                cuplan.handle,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &workspace_size,
                sizeof(workspace_size));
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    struct product_plan* plan_struct = new (std::nothrow) struct product_plan;
    if (!plan_struct) return TAPP_ERR_OUT_OF_MEMORY; // cuplan still owns the plan and destroys it
    plan_struct->plan = cuplan.release();
    plan_struct->type_D = info_D->type;
    plan_struct->conj_scalars = flip;
    plan_struct->workspace_size = workspace_size;
    plan_struct->device = exec_struct->device;
    *plan = (TAPP_tensor_product) plan_struct;

    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    if (!plan) return TAPP_ERR_NULL_POINTER;

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
    struct executor* exec_struct = (struct executor*) exec;

    if (plan_struct->device != exec_struct->device) return TAPP_ERR_PLAN_WRONG_DEVICE;

    // C is only read when beta is not zero
    if (!C && !is_zero_scalar(plan_struct->type_D, beta)) return TAPP_ERR_NULL_POINTER;

    if (!is_aligned(A) || !is_aligned(B) || !is_aligned(C) || !is_aligned(D)) return TAPP_ERR_MISALIGNED;

    // op_D is folded into op_A, op_B and op_C, so a conjugated D also conjugates the scalars
    std::complex<double> alpha_buf, beta_buf; // Room for a scalar of any type
    load_scalar(plan_struct->type_D, alpha, plan_struct->conj_scalars, &alpha_buf);
    load_scalar(plan_struct->type_D, beta, plan_struct->conj_scalars, &beta_buf);

    cutensorHandle_t libhandle = exec_struct->libhandle;

    const cudaStream_t stream = exec_struct->stream;
    cutensorStatus_t err;

    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    if (plan_struct->workspace_size > exec_struct->workspace_capacity)
    {
        if (exec_struct->workspace) cudaFreeAsync(exec_struct->workspace, stream);
        exec_struct->workspace = nullptr;
        exec_struct->workspace_capacity = 0;
        cudaError_t cerr = cudaMallocAsync(&exec_struct->workspace, plan_struct->workspace_size, stream);
        if (cerr != cudaSuccess) return tapp_error(cerr);
        exec_struct->workspace_capacity = plan_struct->workspace_size;
    }

    err = cutensorContract(libhandle,
                plan_struct->plan,
                &alpha_buf, A, B,
                &beta_buf,  C ? C : D, D, // cuTENSOR refuses NULL even when beta is zero and C is not read, D has the same layout
                exec_struct->workspace, plan_struct->workspace_size, stream);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    return create_status(stream, exec_struct->device, status);
}
