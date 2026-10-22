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

    cutensorComputeDescriptor_t cuprec;
    if (!translate_prectype(prec, info_C->type, cuprec)) return TAPP_ERR_UNSUPPORTED_PRECISION;

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

    cutensorStatus_t err;

    tensor_descriptor descA;
    err = create_descriptor(libhandle, info_A, descA);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    // B has the layout of C, so B needs no descriptor of its own
    tensor_descriptor descC;
    err = create_descriptor(libhandle, info_C, descC);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    plan_preference plan_pref;
    err = cutensorCreatePlanPreference(libhandle,
                &plan_pref.handle,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    // cuTENSOR computes D = alpha op(A) + gamma op(C) and requires the same descriptor and modes for its C and D: TAPP B
    // takes the place of its C and TAPP C that of its D, both with the layout of TAPP C
    operation_descriptor desc;
    err = cutensorCreateElementwiseBinary(libhandle,
                &desc.handle,
                descA.handle, cuidx_A.data(), cuop_A,
                descC.handle, cuidx_C.data(), cuop_B,
                descC.handle, cuidx_C.data(),
                CUTENSOR_OP_ADD,
                cuprec);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    cutensorDataType_t scalar_type;
    err = cutensorOperationDescriptorGetAttribute(libhandle,
                desc.handle,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalar_type,
                sizeof(scalar_type));
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    // execute passes alpha and beta in the datatype of C, so cuTENSOR must expect that type
    if (scalar_type != info_C->type) return TAPP_ERR_UNSUPPORTED_PRECISION;

    plan_owner cuplan;
    err = cutensorCreatePlan(libhandle, &cuplan.handle, desc.handle, plan_pref.handle, 0);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    struct permute_plan* plan_struct = new (std::nothrow) struct permute_plan;
    if (!plan_struct) return TAPP_ERR_OUT_OF_MEMORY;   // cuplan still owns the plan and destroys it
    plan_struct->plan = cuplan.release();
    plan_struct->type_C = info_C->type;
    plan_struct->conj_scalars = flip;
    plan_struct->device = exec_struct->device;
    *plan = (TAPP_tensor_permute) plan_struct;

    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_tensor_permute(TAPP_tensor_permute plan)
{
    if (!plan) return TAPP_ERR_NULL_POINTER;

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
    struct executor* exec_struct = (struct executor*) exec;

    if (plan_struct->device != exec_struct->device) return TAPP_ERR_PLAN_WRONG_DEVICE;

    // B is only read when beta is not zero
    if (!B && !is_zero_scalar(plan_struct->type_C, beta)) return TAPP_ERR_NULL_POINTER;

    if (!is_aligned(A) || !is_aligned(B) || !is_aligned(C)) return TAPP_ERR_MISALIGNED;

    // op_C is folded into op_A and op_B, so a conjugated C also conjugates the scalars
    std::complex<double> alpha_buf, beta_buf; // Room for a scalar of any type
    load_scalar(plan_struct->type_C, alpha, plan_struct->conj_scalars, &alpha_buf);
    load_scalar(plan_struct->type_C, beta, plan_struct->conj_scalars, &beta_buf);

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
