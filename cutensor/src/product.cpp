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
    // cuTENSOR computes D in the layout of C, so they must have the same structure
    if (!same_structure((struct tensor_info*)C, idx_C, (struct tensor_info*)D, idx_D)) return TAPP_ERR_C_D_MISMATCH;

    struct product_plan* plan_struct = new struct product_plan;
    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    std::vector<int32_t> cuidx_A = std::vector<int32_t>(idx_A, idx_A + TAPP_get_nmodes(A));
    std::vector<int32_t> cuidx_B = std::vector<int32_t>(idx_B, idx_B + TAPP_get_nmodes(B));
    std::vector<int32_t> cuidx_C = std::vector<int32_t>(idx_C, idx_C + TAPP_get_nmodes(C));
    std::vector<int32_t> cuidx_D = std::vector<int32_t>(idx_D, idx_D + TAPP_get_nmodes(D));

    cudaError_t cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);
    
    // Create tensor descriptor for A
    struct tensor_info* info_A_ptr = (struct tensor_info*)A;
    cutensorTensorDescriptor_t descA;
    cutensorStatus_t err = cutensorCreateTensorDescriptor(libhandle,
                &descA,
                info_A_ptr->nmode,
                info_A_ptr->extents,
                info_A_ptr->strides,
                translate_datatype(info_A_ptr->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    struct tensor_info* info_B_ptr = (struct tensor_info*)B;
    cutensorTensorDescriptor_t descB;
    err = cutensorCreateTensorDescriptor(libhandle,
                &descB,
                info_B_ptr->nmode,
                info_B_ptr->extents,
                info_B_ptr->strides,
                translate_datatype(info_B_ptr->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    struct tensor_info* info_C_ptr = (struct tensor_info*)C;
    cutensorTensorDescriptor_t descC;
    err = cutensorCreateTensorDescriptor(libhandle,
                &descC,
                info_C_ptr->nmode,
                info_C_ptr->extents,
                info_C_ptr->strides,
                translate_datatype(info_C_ptr->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    struct tensor_info* info_D_ptr = (struct tensor_info*)D;
    cutensorTensorDescriptor_t descD;
    err = cutensorCreateTensorDescriptor(libhandle,
                &descD,
                info_D_ptr->nmode,
                info_D_ptr->extents,
                info_D_ptr->strides,
                translate_datatype(info_D_ptr->type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);

    // Resources are null-initialized so the fail: path only releases what was created.
    cutensorOperationDescriptor_t contraction_desc = nullptr;
    cutensorOperationDescriptor_t permutation_desc = nullptr;
    cutensorPlanPreference_t plan_pref = nullptr;
    plan_struct->contraction_plan = nullptr;
    plan_struct->permutation_plan = nullptr;

    cutensorDataType_t scalarType;
    uint64_t workspace_size_estimate = 0;

    err = cutensorCreateContraction(libhandle,
                &contraction_desc,
                descA, cuidx_A.data(), translate_operator(op_A),
                descB, cuidx_B.data(), translate_operator(op_B),
                descC, cuidx_C.data(), translate_operator(op_C),
                descD, cuidx_D.data(),
                translate_prectype(prec, ((struct tensor_info*)D)->type));
    if (err != CUTENSOR_STATUS_SUCCESS) { contraction_desc = nullptr; goto fail; }

    err = cutensorOperationDescriptorGetAttribute(libhandle,
                contraction_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType));
    if (err != CUTENSOR_STATUS_SUCCESS) goto fail;

    assert(scalarType == translate_datatype(((struct tensor_info*)D)->type));

    err = cutensorCreatePermutation(libhandle,
        &permutation_desc,
        descD, cuidx_D.data(), translate_operator(op_D),
        descD, cuidx_D.data(),
        translate_prectype(prec, ((tensor_info*)D)->type));
    if (err != CUTENSOR_STATUS_SUCCESS) { permutation_desc = nullptr; goto fail; }

    err = cutensorOperationDescriptorGetAttribute(libhandle,
                permutation_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType));
    if (err != CUTENSOR_STATUS_SUCCESS) goto fail;

    assert(scalarType == translate_datatype(((struct tensor_info*)D)->type));

    err = cutensorCreatePlanPreference(
                libhandle,
                &plan_pref,
                CUTENSOR_ALGO_DEFAULT,
                CUTENSOR_JIT_MODE_NONE);
    if (err != CUTENSOR_STATUS_SUCCESS) { plan_pref = nullptr; goto fail; }

    err = cutensorEstimateWorkspaceSize(libhandle,
                contraction_desc,
                plan_pref,
                CUTENSOR_WORKSPACE_DEFAULT,
                &workspace_size_estimate);
    if (err != CUTENSOR_STATUS_SUCCESS) goto fail;

    plan_struct->contraction_plan = new cutensorPlan_t;
    err = cutensorCreatePlan(libhandle,
                plan_struct->contraction_plan,
                contraction_desc,
                plan_pref,
                workspace_size_estimate);
    if (err != CUTENSOR_STATUS_SUCCESS) { delete plan_struct->contraction_plan; plan_struct->contraction_plan = nullptr; goto fail; }

    plan_struct->permutation_plan = new cutensorPlan_t;
    err = cutensorCreatePlan(libhandle,
        plan_struct->permutation_plan,
        permutation_desc,
        plan_pref,
        workspace_size_estimate
    );
    if (err != CUTENSOR_STATUS_SUCCESS) { delete plan_struct->permutation_plan; plan_struct->permutation_plan = nullptr; goto fail; }

    plan_struct->copy_size_D = ((struct tensor_info*)D)->copy_size;
    plan_struct->type_D = ((struct tensor_info*)D)->type;
    plan_struct->op_D = op_D;

    *plan = (TAPP_tensor_product) plan_struct;
    err = cutensorDestroyOperationDescriptor(contraction_desc);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    err = cutensorDestroyOperationDescriptor(permutation_desc);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    err = cutensorDestroyPlanPreference(plan_pref);
    return tapp_error(err);

fail:
    if (plan_struct->contraction_plan) { cutensorDestroyPlan(*plan_struct->contraction_plan); delete plan_struct->contraction_plan; }
    if (plan_struct->permutation_plan) { cutensorDestroyPlan(*plan_struct->permutation_plan); delete plan_struct->permutation_plan; }
    if (contraction_desc) cutensorDestroyOperationDescriptor(contraction_desc);
    if (permutation_desc) cutensorDestroyOperationDescriptor(permutation_desc);
    if (plan_pref) cutensorDestroyPlanPreference(plan_pref);
    delete plan_struct;
    return tapp_error(err);
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    struct product_plan* plan_struct = (struct product_plan*) plan;
    cutensorStatus_t err;
    err = cutensorDestroyPlan(*plan_struct->contraction_plan);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    delete plan_struct->contraction_plan;
    err = cutensorDestroyPlan(*plan_struct->permutation_plan);
    if (err != CUTENSOR_STATUS_SUCCESS) return tapp_error(err);
    delete plan_struct->permutation_plan;
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
    struct executor* exec_struct = (struct executor*) exec;
    cutensorHandle_t libhandle = exec_struct->libhandle;

    const bool do_permutation = ( ((struct product_plan*)plan)->op_D != TAPP_IDENTITY );
    const cudaStream_t stream = exec_struct->stream;
    cudaError_t cerr;
    cutensorStatus_t err = CUTENSOR_STATUS_SUCCESS;
    TAPP_error result = TAPP_SUCCESS;

    cerr = cudaSetDevice(exec_struct->device);
    if (cerr != cudaSuccess) return tapp_error(cerr);

    // Working pointers (possibly offset-adjusted) and the allocations to free.
    // The *_alloc pointers are only set when this call owns the buffer, so the
    // cleanup path never frees the caller's device memory.
    void *A_d = nullptr, *B_d = nullptr, *C_d = nullptr, *D_d = nullptr;
    void *A_d_alloc = nullptr, *B_d_alloc = nullptr, *C_d_alloc = nullptr, *D_d_alloc = nullptr;
    void *E_d = nullptr, *E_d_alloc = nullptr;
    void *contraction_work = nullptr;
    void *perm_scalar_ptr = nullptr;
    cutensorPlan_t* contraction_plan = ((struct product_plan*) plan)->contraction_plan;
    uint64_t contraction_actual_workspace_size = 0;
    void* contraction_output = nullptr;

    if (do_permutation) {
        cerr = cudaMallocAsync(&E_d_alloc, ((struct product_plan*)plan)->copy_size_D, stream);
        if (cerr != cudaSuccess) { result = tapp_error(cerr); goto cleanup; }
        E_d = E_d_alloc;
    }

    A_d = (void*)A;
    B_d = (void*)B;
    C_d = (void*)C;
    D_d = (void*)D;

    err = cutensorPlanGetAttribute(libhandle,
                *contraction_plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &contraction_actual_workspace_size,
                sizeof(contraction_actual_workspace_size));
    if (err != CUTENSOR_STATUS_SUCCESS) { result = tapp_error(err); goto cleanup; }

    unsigned int flags;
    cerr = cudaStreamGetFlags(stream, &flags);
    if(cerr!=cudaSuccess) tapp_error(cerr);

    contraction_actual_workspace_size = std::max(contraction_actual_workspace_size, uint64_t(128 * 1024 * 1024)); // 128 MiB recomended minimum size https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcontrac
    cerr = cudaMallocAsync(&contraction_work, contraction_actual_workspace_size, stream);
    if (cerr != cudaSuccess) { result = tapp_error(cerr); goto cleanup; }
    assert(uintptr_t(contraction_work) % 128 == 0);

    contraction_output = do_permutation ? E_d : D_d;
    err = cutensorContract(libhandle,
                *contraction_plan,
                alpha, A_d, B_d,
                beta,  C_d, contraction_output,
                contraction_work, contraction_actual_workspace_size, stream);
    if (err != CUTENSOR_STATUS_SUCCESS) { result = tapp_error(err); goto cleanup; }

    if (do_permutation)
    {
        cutensorPlan_t* permutation_plan = ((struct product_plan*) plan)->permutation_plan;

        if (((struct product_plan*)plan)->type_D == TAPP_F32)
        {
            perm_scalar_ptr = malloc(sizeof(float));
            *(float*)perm_scalar_ptr = 1.0f;
        }
        else if (((struct product_plan*)plan)->type_D == TAPP_F64)
        {
            perm_scalar_ptr = malloc(sizeof(double));
            *(double*)perm_scalar_ptr = 1.0;
        }
        else if (((struct product_plan*)plan)->type_D == TAPP_C32)
        {
            perm_scalar_ptr = malloc(sizeof(std::complex<float>));
            *(std::complex<float>*)perm_scalar_ptr = 1.0f;
        }
        else if (((struct product_plan*)plan)->type_D == TAPP_C64)
        {
            perm_scalar_ptr = malloc(sizeof(std::complex<double>));
            *(std::complex<double>*)perm_scalar_ptr = 1.0;
        }

        err = cutensorPermute(libhandle,
                    *permutation_plan,
                    perm_scalar_ptr,
                    E_d,
                    D_d,
                    stream);
        if (err != CUTENSOR_STATUS_SUCCESS) { result = tapp_error(err); goto cleanup; }
        free(perm_scalar_ptr);
        perm_scalar_ptr = nullptr;
    }

    result = create_status(stream, exec_struct->device, status);
    if (!TAPP_check_success(result)) goto cleanup;

cleanup:
    if (perm_scalar_ptr) free(perm_scalar_ptr);
    if (A_d_alloc) { cerr = cudaFreeAsync(A_d_alloc, stream); if (cerr != cudaSuccess && TAPP_check_success(result)) result = tapp_error(cerr); }
    if (B_d_alloc) { cerr = cudaFreeAsync(B_d_alloc, stream); if (cerr != cudaSuccess && TAPP_check_success(result)) result = tapp_error(cerr); }
    if (C_d_alloc) { cerr = cudaFreeAsync(C_d_alloc, stream); if (cerr != cudaSuccess && TAPP_check_success(result)) result = tapp_error(cerr); }
    if (D_d_alloc) { cerr = cudaFreeAsync(D_d_alloc, stream); if (cerr != cudaSuccess && TAPP_check_success(result)) result = tapp_error(cerr); }
    if (E_d_alloc) { cerr = cudaFreeAsync(E_d_alloc, stream); if (cerr != cudaSuccess && TAPP_check_success(result)) result = tapp_error(cerr); }
    if (contraction_work) { cerr = cudaFreeAsync(contraction_work, stream); if (cerr != cudaSuccess && TAPP_check_success(result)) result = tapp_error(cerr); }
    return result;
}
