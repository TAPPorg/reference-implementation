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
    if ((info_A->nmode > 0 && !idx_A) || (info_B->nmode > 0 && !idx_B) || (info_C->nmode > 0 && !idx_C))
    {
        return TAPP_ERR_NULL_POINTER;
    }
    if (info_A->type != info_C->type || info_B->type != info_C->type) return TAPP_ERR_INVALID_DATATYPE;

    // cuTENSOR computes C in the layout of B, so they must have the same structure
    if (!same_structure(info_B, idx_B, info_C, idx_C)) return TAPP_ERR_B_C_MISMATCH;

    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    std::vector<int32_t> cuidx_A = std::vector<int32_t>(idx_A, idx_A + info_A->nmode);
    std::vector<int32_t> cuidx_C = std::vector<int32_t>(idx_C, idx_C + info_C->nmode);

    // conj(alpha op_A(A) + beta op_B(B)) = conj(alpha) op_A'(A) + conj(beta) op_B'(B), with op_A', op_B' flipped
    const bool flip = (op_C == TAPP_CONJUGATE);
    const TAPP_element_op op_A_eff = ((op_A == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY;
    const TAPP_element_op op_B_eff = ((op_B == TAPP_CONJUGATE) != flip) ? TAPP_CONJUGATE : TAPP_IDENTITY;

    cudaError_t cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    // Resources are null-initialized so cleanup only releases what was created. They are declared before the first
    // goto, which may not jump past an initialization
    cutensorTensorDescriptor_t descA = nullptr;
    cutensorTensorDescriptor_t descC = nullptr;
    cutensorOperationDescriptor_t desc = nullptr;
    cutensorPlanPreference_t plan_pref = nullptr;
    cutensorPlan_t cuplan = nullptr;
    cutensorStatus_t err;

    err = cutensorCreateTensorDescriptor(libhandle,
                &descA,
                info_A->nmode,
                info_A->extents,
                info_A->strides,
                translate_datatype(info_A->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // B has the layout of C, so B needs no descriptor of its own
    err = cutensorCreateTensorDescriptor(libhandle,
                &descC,
                info_C->nmode,
                info_C->extents,
                info_C->strides,
                translate_datatype(info_C->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    err = cutensorCreatePlanPreference(libhandle,
                &plan_pref,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

    // cuTENSOR computes D = alpha op(A) + gamma op(C) and requires the same descriptor and modes for its C and D: TAPP B
    // takes the place of its C and TAPP C that of its D, both with the layout of TAPP C
    err = cutensorCreateElementwiseBinary(libhandle,
                &desc,
                descA, cuidx_A.data(), translate_operator(op_A_eff),
                descC, cuidx_C.data(), translate_operator(op_B_eff),
                descC, cuidx_C.data(),
                CUTENSOR_OP_ADD,
                translate_prectype(prec, info_C->type));
    if (err != CUTENSOR_STATUS_SUCCESS) goto cleanup;

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
    return tapp_error(err);
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

    // op_C is folded into op_A and op_B, so a conjugated C also conjugates the scalars
    std::complex<double> alpha_buf, beta_buf; // Room for a scalar of any type
    load_scalar(plan_struct->type_C, alpha, plan_struct->conj_scalars, &alpha_buf);
    load_scalar(plan_struct->type_C, beta, plan_struct->conj_scalars, &beta_buf);

    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    const cudaStream_t stream = exec_struct->stream;
    cudaError_t cerr;
    cutensorStatus_t err = CUTENSOR_STATUS_SUCCESS;
    TAPP_error result = TAPP_SUCCESS;

    cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    err = cutensorElementwiseBinaryExecute(
        libhandle,
        plan_struct->plan,
        &alpha_buf,
        A,
        &beta_buf,
        B,
        C,
        stream
    );
    if (err != CUTENSOR_STATUS_SUCCESS) { return tapp_error(err); }

    // CUDA Event
    result = create_status(stream, exec_struct->device, status);

    return result;
}
