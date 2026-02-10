#include "../include/product.h"
#include "../include/attributes.h"

int64_t compute_index(const int64_t* coordinates, int nmode, const int64_t* strides);
void increment_coordinates(int64_t* coordinates, int nmode, const int64_t* extents);
cutensorOperator_t translate_operator(TAPP_element_op op);

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
    struct product_plan* plan_struct = new struct product_plan;
    plan_struct->handle = ((cutensorHandle_t*) handle);
    struct handle* handle_struct = (struct handle*) plan_struct->handle;
    std::vector<int32_t> cuidx_A = std::vector<int32_t>(idx_A, idx_A + TAPP_get_nmodes(A));
    std::vector<int32_t> cuidx_B = std::vector<int32_t>(idx_B, idx_B + TAPP_get_nmodes(B));
    std::vector<int32_t> cuidx_C = std::vector<int32_t>(idx_C, idx_C + TAPP_get_nmodes(C));
    std::vector<int32_t> cuidx_D = std::vector<int32_t>(idx_D, idx_D + TAPP_get_nmodes(D));

    cutensorStatus_t err;
    cutensorOperationDescriptor_t contraction_desc;
    err = cutensorCreateContraction(*handle_struct->libhandle, 
                &contraction_desc,
                *((struct tensor_info*)A)->desc, cuidx_A.data(), translate_operator(op_A),
                *((struct tensor_info*)B)->desc, cuidx_B.data(), translate_operator(op_B),
                *((struct tensor_info*)C)->desc, cuidx_C.data(), translate_operator(op_C),
                *((struct tensor_info*)D)->desc, cuidx_D.data(),
                translate_prectype(prec, ((struct tensor_info*)D)->type));
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    cutensorDataType_t scalarType;
    err = cutensorOperationDescriptorGetAttribute(*handle_struct->libhandle,
                contraction_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType));
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    assert(scalarType == translate_datatype(((struct tensor_info*)D)->type));

    cutensorOperationDescriptor_t permutation_desc;
    err = cutensorCreatePermutation(*handle_struct->libhandle,
        &permutation_desc,
        *((struct tensor_info*)D)->desc, cuidx_D.data(), translate_operator(op_D),
        *((struct tensor_info*)D)->desc, cuidx_D.data(),
        translate_prectype(prec, ((tensor_info*)D)->type));
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    err = cutensorOperationDescriptorGetAttribute(*handle_struct->libhandle,
                permutation_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType));
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    assert(scalarType == translate_datatype(((struct tensor_info*)D)->type));

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t plan_pref;
    err = cutensorCreatePlanPreference(
                *handle_struct->libhandle,
                &plan_pref,
                algo,
                CUTENSOR_JIT_MODE_NONE);
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    uint64_t workspace_size_estimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    cutensorEstimateWorkspaceSize(*handle_struct->libhandle,
                contraction_desc,
                plan_pref,
                workspacePref,
                &workspace_size_estimate);

    plan_struct->contraction_plan = new cutensorPlan_t;
    err = cutensorCreatePlan(*handle_struct->libhandle,
                plan_struct->contraction_plan,
                contraction_desc,
                plan_pref,
                workspace_size_estimate);
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    plan_struct->permutation_plan = new cutensorPlan_t;
    err = cutensorCreatePlan(*handle_struct->libhandle,
        plan_struct->permutation_plan,
        permutation_desc,
        plan_pref,
        workspace_size_estimate
    );
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    plan_struct->data_offset_A = ((struct tensor_info*)A)->data_offset;
    plan_struct->copy_size_A = ((struct tensor_info*)A)->copy_size;
    plan_struct->data_offset_B = ((struct tensor_info*)B)->data_offset;
    plan_struct->copy_size_B = ((struct tensor_info*)B)->copy_size;
    plan_struct->data_offset_C = ((struct tensor_info*)C)->data_offset;
    plan_struct->copy_size_C = ((struct tensor_info*)C)->copy_size;
    plan_struct->data_offset_D = ((struct tensor_info*)D)->data_offset;
    plan_struct->copy_size_D = ((struct tensor_info*)D)->copy_size;
    plan_struct->sections_D = 1;
    plan_struct->section_size_D = 1;
    plan_struct->sections_nmode_D = 0;
    plan_struct->section_strides_D = new int64_t[TAPP_get_nmodes(D)];
    plan_struct->section_extents_D = new int64_t[TAPP_get_nmodes(D)];
    plan_struct->type_D = ((struct tensor_info*)D)->type;
    plan_struct->op_D = op_D;
    int64_t sorted_strides_D[TAPP_get_nmodes(D)];
    memcpy(sorted_strides_D, ((struct tensor_info*)D)->strides, TAPP_get_nmodes(D) * sizeof(int64_t));
    auto compare = [](int64_t a, int64_t b) { return std::abs(a) < std::abs(b); };
    std::sort(sorted_strides_D, sorted_strides_D + TAPP_get_nmodes(D), compare);
    for (int i = 0; i < TAPP_get_nmodes(D); i++)
    {
        for (int j = 0; j < TAPP_get_nmodes(D); j++)
        {
            if (((struct tensor_info*)D)->strides[j] == sorted_strides_D[i])
            {
                if (std::abs(sorted_strides_D[i]) == plan_struct->section_size_D)
                {
                    plan_struct->section_size_D *= std::abs(((struct tensor_info*)D)->extents[i]);
                }
                else if (((struct tensor_info*)D)->extents[j] != 1) // if extent = 0 then stride will never be used i.e. no need for section, even if stride would create section
                {
                    plan_struct->sections_D *= ((struct tensor_info*)D)->extents[j];
                    plan_struct->section_extents_D[plan_struct->sections_nmode_D] = ((struct tensor_info*)D)->extents[j];
                    plan_struct->section_strides_D[plan_struct->sections_nmode_D] = ((struct tensor_info*)D)->strides[j];
                    plan_struct->sections_nmode_D++;
                }
                break;
            }
        }
    }
    plan_struct->section_size_D *= sizeof_datatype(((struct tensor_info*)D)->type);
    *plan = (TAPP_tensor_product) plan_struct;
    err = cutensorDestroyOperationDescriptor(contraction_desc);
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);
    err = cutensorDestroyOperationDescriptor(permutation_desc);
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);
    cutensorDestroyPlanPreference(plan_pref);
    return pack_error(0, err); 
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    struct product_plan* plan_struct = (struct product_plan*) plan;
    cutensorStatus_t err;
    err = cutensorDestroyPlan(*plan_struct->contraction_plan);
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);
    delete plan_struct->contraction_plan;
    err = cutensorDestroyPlan(*plan_struct->permutation_plan);
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);
    delete plan_struct->permutation_plan;
    delete[] plan_struct->section_strides_D;
    delete[] plan_struct->section_extents_D;
    delete plan_struct;
    return pack_error(0, err); 
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
    void *A_d, *B_d, *C_d, *D_d;
    struct handle* handle_struct = (struct handle*) ((struct product_plan*) plan)->handle;
    bool use_device_memory;
    TAPP_attr_get((TAPP_handle)handle_struct, ATTR_KEY_USE_DEVICE_MEMORY, (void*)&use_device_memory);
    const bool do_permutation = ( ((struct product_plan*)plan)->op_D != TAPP_IDENTITY );
    cudaError_t cerr;

    void *E_d = nullptr;
    if (do_permutation) {
        cerr = cudaMallocAsync((void**)&E_d, ((struct product_plan*)plan)->copy_size_D, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
    }
    
    if (use_device_memory)
    {
        A_d = (void*)A;
        B_d = (void*)B;
        C_d = (void*)C;
        D_d = (void*)D;
    }
    else
    {
        cerr = cudaMallocAsync((void**)&A_d, ((struct product_plan*)plan)->copy_size_A, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMallocAsync((void**)&B_d, ((struct product_plan*)plan)->copy_size_B, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMallocAsync((void**)&C_d, ((struct product_plan*)plan)->copy_size_C, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMallocAsync((void**)&D_d, ((struct product_plan*)plan)->copy_size_D, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMemcpyAsync(A_d, (void*)((intptr_t)A + ((struct product_plan*)plan)->data_offset_A), ((struct product_plan*)plan)->copy_size_A, cudaMemcpyHostToDevice, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMemcpyAsync(B_d, (void*)((intptr_t)B + ((struct product_plan*)plan)->data_offset_B), ((struct product_plan*)plan)->copy_size_B, cudaMemcpyHostToDevice, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        cerr = cudaMemcpyAsync(C_d, (void*)((intptr_t)C + ((struct product_plan*)plan)->data_offset_C), ((struct product_plan*)plan)->copy_size_C, cudaMemcpyHostToDevice, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        A_d = (void*)((intptr_t)A_d + ((struct product_plan*)plan)->data_offset_A);
        B_d = (void*)((intptr_t)B_d + ((struct product_plan*)plan)->data_offset_B);
        C_d = (void*)((intptr_t)C_d + ((struct product_plan*)plan)->data_offset_C);
        D_d = (void*)((intptr_t)D_d + ((struct product_plan*)plan)->data_offset_D);
        if (do_permutation) {
            E_d = (void*)((intptr_t)E_d + ((struct product_plan*)plan)->data_offset_D);
        }
        assert(uintptr_t(A_d) % 128 == 0);
        assert(uintptr_t(B_d) % 128 == 0);
        assert(uintptr_t(C_d) % 128 == 0);
        assert(uintptr_t(D_d) % 128 == 0);
    }
    cutensorPlan_t* contraction_plan = ((struct product_plan*) plan)->contraction_plan;
    uint64_t contraction_actual_workspace_size = 0;
    cutensorStatus_t err;
    err = cutensorPlanGetAttribute(*handle_struct->libhandle,
                *contraction_plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &contraction_actual_workspace_size,
                sizeof(contraction_actual_workspace_size));
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    // TODO Recommended minimum 128 MB workspace 
    // https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html#cutensorcontract
    // contraction_actual_workspace_size = std::max(contraction_actual_workspace_size, uint64_t(128 * 1024 * 1024)); // 128 MiB
    void *contraction_work = nullptr;
    if (contraction_actual_workspace_size > 0)
    {
        cerr = cudaMallocAsync(&contraction_work, contraction_actual_workspace_size, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
        assert(uintptr_t(contraction_work) % 128 == 0);
    }

    void* contraction_output = do_permutation ? E_d : D_d;
    err = cutensorContract(*handle_struct->libhandle,
                *contraction_plan,
                alpha, A_d, B_d,
                beta,  C_d, contraction_output, 
                contraction_work, contraction_actual_workspace_size, *(cudaStream_t*)exec);
    if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);

    if (do_permutation)
    {
        cutensorPlan_t* permutation_plan = ((struct product_plan*) plan)->permutation_plan;
        void* perm_scalar_ptr = NULL;

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

        err = cutensorPermute(*handle_struct->libhandle,
                    *permutation_plan,
                    perm_scalar_ptr,
                    E_d,
                    D_d,
                    *(cudaStream_t*)exec);
        if (err != CUTENSOR_STATUS_SUCCESS) return pack_error(0, err);
        free(perm_scalar_ptr);
    }

    if (!use_device_memory)
    {
        int64_t section_coordinates_D[((struct product_plan*)plan)->sections_nmode_D];
        for (size_t i = 0; i < ((struct product_plan*)plan)->sections_nmode_D; i++)
        {
            section_coordinates_D[i] = 0;
        }

        for (size_t i = 0; i < ((struct product_plan*)plan)->sections_D; i++)
        {
            int64_t index = compute_index(section_coordinates_D, ((struct product_plan*)plan)->sections_nmode_D, ((struct product_plan*)plan)->section_strides_D);
            cerr = cudaMemcpyAsync((void*)((intptr_t)D + index * sizeof_datatype(((struct product_plan*)plan)->type_D)), 
                (void*)((intptr_t)D_d + index * sizeof_datatype(((struct product_plan*)plan)->type_D)), 
                ((struct product_plan*)plan)->section_size_D, cudaMemcpyDeviceToHost, *(cudaStream_t*)exec);
            if (cerr != cudaSuccess) return pack_error(0, cerr);
            increment_coordinates(section_coordinates_D, ((struct product_plan*)plan)->sections_nmode_D, ((struct product_plan*)plan)->section_extents_D);
        }

        A_d = (void*)((intptr_t)A_d - ((struct product_plan*)plan)->data_offset_A);
        B_d = (void*)((intptr_t)B_d - ((struct product_plan*)plan)->data_offset_B);
        C_d = (void*)((intptr_t)C_d - ((struct product_plan*)plan)->data_offset_C);
        D_d = (void*)((intptr_t)D_d - ((struct product_plan*)plan)->data_offset_D);

        if (A_d) { 
            cerr = cudaFreeAsync(A_d, *(cudaStream_t*)exec);
            if (cerr != cudaSuccess) return pack_error(0, cerr);
        }
        if (B_d) {
            cerr = cudaFreeAsync(B_d, *(cudaStream_t*)exec);
            if (cerr != cudaSuccess) return pack_error(0, cerr);
        }
        if (C_d) { 
            cerr = cudaFreeAsync(C_d, *(cudaStream_t*)exec);
            if (cerr != cudaSuccess) return pack_error(0, cerr);
        }
        if (D_d) {
            cerr = cudaFreeAsync(D_d, *(cudaStream_t*)exec);
            if (cerr != cudaSuccess) return pack_error(0, cerr);
        }
    }

    if (E_d)
    {
        if (!use_device_memory)
        {
            E_d = (void*)((intptr_t)E_d - ((struct product_plan*)plan)->data_offset_D);
        }
        cerr = cudaFreeAsync(E_d, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
    }
    if (contraction_work) {
        cerr = cudaFreeAsync(contraction_work, *(cudaStream_t*)exec);
        if (cerr != cudaSuccess) return pack_error(0, cerr);
    }

    return pack_error(0, err); 
}

int64_t compute_index(const int64_t* coordinates, int nmode, const int64_t* strides)
{
    int64_t index = 0;
    for (int i = 0; i < nmode; i++)
    {
        index += coordinates[i] * strides[i];
    }
    return index;

}

void increment_coordinates(int64_t* coordinates, int nmode, const int64_t* extents)
{
    if (nmode <= 0)
    {
        return;
    }

    int k = 0;
    do
    {
        coordinates[k] = (coordinates[k] + 1) % extents[k];
        k++;
    } while (coordinates[k - 1] == 0 && k < nmode);
}

cutensorOperator_t translate_operator(TAPP_element_op op)
{
    switch (op)
    {
    case TAPP_IDENTITY:
        return CUTENSOR_OP_IDENTITY;
        break;
    case TAPP_CONJUGATE:
        return CUTENSOR_OP_CONJ;
        break;
    default: // TODO: Default should probably be an error
        return CUTENSOR_OP_IDENTITY;
        break;
    }
}
