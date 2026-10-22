#include "internal.h"

TAPP_error TAPP_create_tensor_permute(TAPP_tensor_permute* plan,
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
                                      TAPP_prectype prec)
{
    // Validate the arguments before creating anything, so these errors need no cleanup
    if (!plan || !exec || !A || !B || !C) return TAPP_ERR_NULL_POINTER;

    const struct tensor_info* info_A = (struct tensor_info*)A;
    const struct tensor_info* info_B = (struct tensor_info*)B;
    const struct tensor_info* info_C = (struct tensor_info*)C;
    if ((info_A->extents.size() > 0 && !idx_A) || (info_B->extents.size() > 0 && !idx_B) || (info_C->extents.size() > 0 && !idx_C))
    {
        return TAPP_ERR_NULL_POINTER;
    }
    if (info_A->type != info_C->type || info_B->type != info_C->type) return TAPP_ERR_INVALID_DATATYPE;
    if (!is_valid_op(op_A) || !is_valid_op(op_B) || !is_valid_op(op_C)) return TAPP_ERR_UNSUPPORTED_ELEMENT_OP;

    // cuTENSOR computes C in the layout of B, so they must have the same structure
    if (!same_structure(info_B, idx_B, info_C, idx_C)) return TAPP_ERR_B_C_MISMATCH;

    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    std::vector<int32_t> cuidx_A = std::vector<int32_t>(idx_A, idx_A + info_A->extents.size());
    std::vector<int32_t> cuidx_C = std::vector<int32_t>(idx_C, idx_C + info_C->extents.size());

    // conj(alpha op_A(A) + beta op_B(B)) = conj(alpha) op_A'(A) + conj(beta) op_B'(B), with op_A', op_B' flipped
    const bool flip = (op_C == TAPP_CONJUGATE);
    cutensorOperator_t cuop_A;
    cutensorOperator_t cuop_B;
    // Cannot fail: the ops were validated above and folding gives IDENTITY or CONJUGATE
    translate_operator(((op_A == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY, cuop_A);
    translate_operator(((op_B == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY, cuop_B);

    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    // Resources are null-initialized so cleanup only releases what was created. They are declared before the first
    // goto, which may not jump past an initialization
    cutensorTensorDescriptor_t descA = nullptr;
    cutensorTensorDescriptor_t descC = nullptr;
    cutensorOperationDescriptor_t desc = nullptr;
    cutensorPlanPreference_t plan_pref = nullptr;
    cutensorPlan_t cuplan = nullptr;
    cutensorStatus_t err;
    TAPP_error result = TAPP_SUCCESS; // Set for errors that are not from cuTENSOR
    cutensorDataType_t scalar_type;

    err = cutensorCreateTensorDescriptor(libhandle,
                &descA,
                info_A->extents.size(),
                info_A->extents.data(),
                info_A->strides.data(),
                info_A->type, kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // B has the layout of C, so B needs no descriptor of its own
    err = cutensorCreateTensorDescriptor(libhandle,
                &descC,
                info_C->extents.size(),
                info_C->extents.data(),
                info_C->strides.data(),
                info_C->type, kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    err = cutensorCreatePlanPreference(libhandle,
                &plan_pref,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    cutensorComputeDescriptor_t cuprec;
    if (!translate_prectype(prec, info_C->type, cuprec))
    {
        result = TAPP_ERR_UNSUPPORTED_PRECISION;
        goto cleanup; // The descriptors already exist
    }
    
    // cuTENSOR computes D = alpha op(A) + gamma op(C) and requires the same descriptor and modes for its C and D: TAPP B
    // takes the place of its C and TAPP C that of its D, both with the layout of TAPP C
    err = cutensorCreateElementwiseBinary(libhandle,
                &desc,
                descA, cuidx_A.data(), cuop_A,
                descC, cuidx_C.data(), cuop_B,
                descC, cuidx_C.data(),
                CUTENSOR_OP_ADD,
                cuprec);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    err = cutensorOperationDescriptorGetAttribute(libhandle,
                desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalar_type,
                sizeof(scalar_type));
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // execute passes alpha and beta in the datatype of C, so cuTENSOR must expect that type
    if (scalar_type != info_C->type)
    {
        result = TAPP_ERR_UNSUPPORTED_PRECISION;
        goto cleanup;
    }

    err = cutensorCreatePlan(libhandle, &cuplan, desc, plan_pref, 0);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    {
        struct permute_plan* plan_struct = new struct permute_plan;
        plan_struct->plan = cuplan;
        plan_struct->type_C = info_C->type;
        plan_struct->conj_scalars = flip;
        *plan = (TAPP_tensor_permute) plan_struct;
        cuplan = nullptr; // Owned by plan_struct now, so cleanup leaves it
    }

    // Reached on success and on failure: the descriptors and the preference are only needed to create the plan
cleanup:
    if (cuplan) cutensorDestroyPlan(cuplan);
    if (desc) cutensorDestroyOperationDescriptor(desc);
    if (plan_pref) cutensorDestroyPlanPreference(plan_pref);
    if (descA) cutensorDestroyTensorDescriptor(descA);
    if (descC) cutensorDestroyTensorDescriptor(descC);
    return result != TAPP_SUCCESS ? result : tapp_error(err);
}

TAPP_error TAPP_destroy_tensor_permute(TAPP_tensor_permute plan)
{
    struct permute_plan* plan_struct = (struct permute_plan*) plan;
    cutensorStatus_t err;
    err = cutensorDestroyPlan(plan_struct->plan);

    delete plan_struct;
    return tapp_error(err);
}

TAPP_error TAPP_execute_permute(TAPP_tensor_permute plan,
                                TAPP_executor exec,
                                TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* beta,
                                const void* B,
                                      void* C)
{
    if (!plan || !exec || !alpha || !beta || !A || !C) return TAPP_ERR_NULL_POINTER;

    struct permute_plan* plan_struct = (struct permute_plan*) plan;

    // B is only read when beta is not zero
    if (!B && !is_zero_scalar(plan_struct->type_C, beta)) return TAPP_ERR_NULL_POINTER;

    if (!is_aligned(A) || !is_aligned(B) || !is_aligned(C)) return TAPP_ERR_MISALIGNED;

    // op_C is folded into op_A and op_B, so a conjugated C also conjugates the scalars
    std::complex<double> alpha_buf, beta_buf; // Room for a scalar of any type
    load_scalar(plan_struct->type_C, alpha, plan_struct->conj_scalars, &alpha_buf);
    load_scalar(plan_struct->type_C, beta, plan_struct->conj_scalars, &beta_buf);

    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    const cudaStream_t stream = exec_struct->stream;
    cutensorStatus_t err;

    device_scope scope(exec_struct->device);
    if (scope.error != cudaSuccess) return tapp_error(scope.error);

    err = cutensorElementwiseBinaryExecute(
        libhandle,
        plan_struct->plan,
        &alpha_buf,
        A,
        &beta_buf,
        B ? B : C, // cuTENSOR refuses NULL even when beta is zero and B is not read, C has the same layout
        C,
        stream
    );
    if (err != CUTENSOR_STATUS_SUCCESS) { return tapp_error(err); }

    // CUDA Event
    return create_status(stream, exec_struct->device, status);
}
