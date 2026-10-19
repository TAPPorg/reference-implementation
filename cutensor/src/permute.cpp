#include "../include/permute.h"
#include "../include/status.h"
#include "../include/executor.h"
#include "../include/util.h"

TAPP_error do_transpose(bool* value, TAPP_datatype type, const void* alpha);
size_t sizieof_TAPP_datatype(TAPP_datatype);

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
    struct permute_plan* plan_struct = new struct permute_plan;
    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t* libhandle = nullptr;
    libhandle = exec_struct->libhandle;

    std::vector<int32_t> cuidx_A = std::vector<int32_t>(idx_A, idx_A + TAPP_get_nmodes(A));
    std::vector<int32_t> cuidx_B = std::vector<int32_t>(idx_B, idx_B + TAPP_get_nmodes(B));
    std::vector<int32_t> cuidx_C = std::vector<int32_t>(idx_C, idx_C + TAPP_get_nmodes(C));

    // Create tensor descriptor for A, B and C
    struct tensor_info* info_A_ptr = (struct tensor_info*)A;
    cutensorTensorDescriptor_t descA;
    cutensorStatus_t err = cutensorCreateTensorDescriptor(*libhandle,
                &descA,
                info_A_ptr->nmode,
                info_A_ptr->extents,
                info_A_ptr->strides,
                translate_datatype(info_A_ptr->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    struct tensor_info* info_B_ptr = (struct tensor_info*)B;
    cutensorTensorDescriptor_t descB;
    err = cutensorCreateTensorDescriptor(*libhandle,
                &descB,
                info_B_ptr->nmode,
                info_B_ptr->extents,
                info_B_ptr->strides,
                translate_datatype(info_B_ptr->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    struct tensor_info* info_C_ptr = (struct tensor_info*)C;
    cutensorTensorDescriptor_t descC;
    err = cutensorCreateTensorDescriptor(*libhandle,
                &descC,
                info_C_ptr->nmode,
                info_C_ptr->extents,
                info_C_ptr->strides,
                translate_datatype(info_C_ptr->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    plan_struct->op_C = op_C;
    plan_struct->copy_size_C = info_C_ptr->copy_size;

    // Resources are null-initialized so the fail: path only releases what was created.
    cutensorOperationDescriptor_t main_permutation_desc = nullptr;
    cutensorOperationDescriptor_t result_permutation_desc = nullptr;
    cutensorPlanPreference_t plan_pref = nullptr;
    plan_struct->main_permutation_plan = nullptr;
    plan_struct->result_permutation_plan = nullptr;

    cutensorDataType_t scalarType;
    cutensorComputeDescriptor_t descCompute;

    // Plan preference
    err = cutensorCreatePlanPreference(
                *libhandle,
                &plan_pref,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (err != CUTENSOR_STATUS_SUCCESS) { plan_pref = nullptr; goto fail; }

    // Permutation
    err = cutensorCreateElementwiseBinary(
                *libhandle,
                &main_permutation_desc,
                descA, cuidx_A.data(), translate_operator(op_A),
                descB, cuidx_B.data(), translate_operator(op_B),
                descC, cuidx_C.data(),
                CUTENSOR_OP_ADD,
                translate_prectype(prec, ((struct tensor_info*)C)->type)

    );
    err = cutensorOperationDescriptorGetAttribute(*libhandle,
                main_permutation_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType));
    if (err != CUTENSOR_STATUS_SUCCESS) goto fail;
    assert(scalarType == translate_datatype(((struct tensor_info*)C)->type));

    plan_struct->main_permutation_plan = new cutensorPlan_t;
    err = cutensorCreatePlan(*libhandle,
        plan_struct->main_permutation_plan,
        main_permutation_desc,
        plan_pref,
        0
    );
    if (err != CUTENSOR_STATUS_SUCCESS) { delete plan_struct->main_permutation_plan; plan_struct->main_permutation_plan = nullptr; goto fail; }

    // Plan for op_C
    if(op_C != TAPP_IDENTITY){
        err = cutensorCreatePermutation(*libhandle,
            &result_permutation_desc,
            descC, cuidx_C.data(), translate_operator(op_C),
            descC, cuidx_C.data(),
            translate_prectype(prec, ((tensor_info*)C)->type));
        if (err != CUTENSOR_STATUS_SUCCESS) { result_permutation_desc = nullptr; goto fail; }

        err = cutensorOperationDescriptorGetAttribute(*libhandle,
                    result_permutation_desc,
                    CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                    (void*)&scalarType,
                    sizeof(scalarType));
        if (err != CUTENSOR_STATUS_SUCCESS) goto fail;

        assert(scalarType == translate_datatype(((struct tensor_info*)C)->type));

        plan_struct->result_permutation_plan = new cutensorPlan_t;
        err = cutensorCreatePlan(*libhandle,
            plan_struct->result_permutation_plan,
            result_permutation_desc,
            plan_pref,
            0
        );
        if (err != CUTENSOR_STATUS_SUCCESS) { delete plan_struct->result_permutation_plan; plan_struct->result_permutation_plan = nullptr; goto fail; }
    }

    *plan = (TAPP_tensor_permute) plan_struct;
    return tapp_error(err);

fail:
    if (plan_struct->main_permutation_plan) { cutensorDestroyPlan(*plan_struct->main_permutation_plan); delete plan_struct->main_permutation_plan; }
    if (plan_struct->result_permutation_plan) { cutensorDestroyPlan(*plan_struct->result_permutation_plan); delete plan_struct->result_permutation_plan; }
    if (main_permutation_desc) cutensorDestroyOperationDescriptor(main_permutation_desc);
    if (result_permutation_desc) cutensorDestroyOperationDescriptor(result_permutation_desc);
    if (plan_pref) cutensorDestroyPlanPreference(plan_pref);
    delete plan_struct;
    return tapp_error(err);

}

TAPP_error TAPP_destroy_tensor_permute(TAPP_tensor_permute plan)
{
    struct permute_plan* plan_struct = (struct permute_plan*) plan;
    cutensorStatus_t err;
    err = cutensorDestroyPlan(*plan_struct->main_permutation_plan);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    delete plan_struct->main_permutation_plan;

    if(plan_struct->op_C != TAPP_IDENTITY){
       err = cutensorDestroyPlan(*plan_struct->result_permutation_plan);
       if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
       delete plan_struct->result_permutation_plan;
    }
    delete plan_struct;
    return tapp_error(err);
}

TAPP_error TAPP_execute_permute(TAPP_tensor_permute plan,
                                TAPP_executor exec,
                                TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* B,
                                const void* beta,
                                      void* C)
{
    struct permute_plan* plan_struct = (struct permute_plan*) plan;

    const bool do_permutation_C = plan_struct->op_C != TAPP_IDENTITY;
    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t* libhandle = nullptr;
    libhandle = exec_struct->libhandle;

    const cudaStream_t stream = *(exec_struct->stream);
    cudaError_t cerr;
    cutensorStatus_t err = CUTENSOR_STATUS_SUCCESS;
    TAPP_error result = TAPP_SUCCESS;

    cerr = cudaSetDevice(*(exec_struct->device));
    if (cerr != cudaSuccess) return tapp_error(cerr);

    void* main_permutation_output = nullptr;
    void* perm_scalar_ptr = nullptr;
    void* C_tmp = nullptr;
    if (do_permutation_C) {
        cerr = cudaMallocAsync(&C_tmp, plan_struct->copy_size_C, stream);
        if (cerr != cudaSuccess) { result = tapp_error(cerr); goto cleanup; }
    }

    // All in one ElementwiseWiseBinary operation
    main_permutation_output = do_permutation_C ? C_tmp : C;
    err = cutensorElementwiseBinaryExecute(
        *libhandle,
        *plan_struct->main_permutation_plan,
        alpha,
        A,
        beta,
        B,
        main_permutation_output,
        stream
    );
    if (err != CUTENSOR_STATUS_SUCCESS) { result = tapp_error(err); goto cleanup; }

    // Apply op_C
    if(do_permutation_C)
    {
        cutensorPlan_t* result_permutation_plan = plan_struct->result_permutation_plan;

        if (plan_struct->type_C == TAPP_F32)
        {
            perm_scalar_ptr = malloc(sizeof(float));
            *(float*)perm_scalar_ptr = 1.0f;
        }
        else if (plan_struct->type_C == TAPP_F64)
        {
            perm_scalar_ptr = malloc(sizeof(double));
            *(double*)perm_scalar_ptr = 1.0;
        }
        else if (plan_struct->type_C == TAPP_C32)
        {
            perm_scalar_ptr = malloc(sizeof(std::complex<float>));
            *(std::complex<float>*)perm_scalar_ptr = 1.0f;
        }
        else if (plan_struct->type_C == TAPP_C64)
        {
            perm_scalar_ptr = malloc(sizeof(std::complex<double>));
            *(std::complex<double>*)perm_scalar_ptr = 1.0;
        }

        err = cutensorPermute(*libhandle,
                    *result_permutation_plan,
                    perm_scalar_ptr,
                    C_tmp,
                    C,
                    stream);
        if (err != CUTENSOR_STATUS_SUCCESS) { result = tapp_error(err); goto cleanup; }
        free(perm_scalar_ptr);
        perm_scalar_ptr = nullptr;
    }

    // CUDA Event
    result = create_status(stream, status);
    if (!TAPP_check_success(result)) goto cleanup;

cleanup:
    if (perm_scalar_ptr) free(perm_scalar_ptr);
    if (C_tmp) { cerr = cudaFreeAsync(C_tmp, stream); if (cerr != cudaSuccess && TAPP_check_success(result)) result = tapp_error(cerr); }
    return result;
}

TAPP_error do_transpose(bool* value, TAPP_datatype type, const void* alpha){
    switch(type){
        case TAPP_F32:
            *value = *((float*) alpha)==0.0f;
            return TAPP_SUCCESS;
            break;
        case TAPP_F64:
            *value = *((double*) alpha)==0.0f;
            return TAPP_SUCCESS;
            break;
        case TAPP_C32:
            *value = *((std::complex<float>*) alpha) ==std::complex<float>(0,0);
            return TAPP_SUCCESS;
            break;
        case TAPP_C64:
            *value = *((std::complex<double>*) alpha)==std::complex<double>(0,0);
            return TAPP_SUCCESS;
            break;
        default: // TODO: Default should probably be an error
            return TAPP_ERR_INVALID_DATATYPE;
            break;
    }
}

size_t sizieof_TAPP_datatype(TAPP_datatype type){
    switch(type){
        case TAPP_F32:
            return sizeof(float);
            break;
        case TAPP_F64:
            return sizeof(double);
            break;
        case TAPP_C32:
            return sizeof(std::complex<float>);
            break;
        case TAPP_C64:
            return sizeof(std::complex<double>);
            break;
    }
   return 0; // return TAPP_error instead?
}
