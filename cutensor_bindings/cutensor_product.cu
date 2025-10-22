#include "../src/tapp/product.h"
#include "cutensor_bind.h"
#include <algorithm>

int64_t compute_index(const int64_t* coordinates, int nmode, const int64_t* strides);
void increment_coordinates(int64_t* coordinates, int nmode, const int64_t* extents);
cutensorOperator_t translate_operator(TAPP_element_op op);

TAPP_EXPORT TAPP_error TAPP_create_tensor_product(TAPP_tensor_product* plan,
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
    cutensor_plan* cuplan = new cutensor_plan;
    cutensorHandle_t cuhandle = *((cutensorHandle_t*) handle);
    std::vector<int32_t> cuidx_A = std::vector<int32_t>(idx_A, idx_A + TAPP_get_nmodes(A));
    std::vector<int32_t> cuidx_B = std::vector<int32_t>(idx_B, idx_B + TAPP_get_nmodes(B));
    std::vector<int32_t> cuidx_C = std::vector<int32_t>(idx_C, idx_C + TAPP_get_nmodes(C));
    std::vector<int32_t> cuidx_D = std::vector<int32_t>(idx_D, idx_D + TAPP_get_nmodes(D));

    cutensorOperationDescriptor_t contraction_desc;
    HANDLE_ERROR(cutensorCreateContraction(cuhandle, 
                &contraction_desc,
                *((cutensor_info*)A)->desc, cuidx_A.data(), translate_operator(op_A),
                *((cutensor_info*)B)->desc, cuidx_B.data(), translate_operator(op_B),
                *((cutensor_info*)C)->desc, cuidx_C.data(), translate_operator(op_C),
                *((cutensor_info*)D)->desc, cuidx_D.data(),
                translate_prectype(prec)));

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(cuhandle,
                contraction_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType)));

    assert(scalarType == translate_datatype(((cutensor_info*)D)->type));

    cutensorOperationDescriptor_t permutation_desc;
    HANDLE_ERROR(cutensorCreatePermutation(cuhandle,
        &permutation_desc,
        *((cutensor_info*)D)->desc, cuidx_D.data(), translate_operator(op_D),
        *((cutensor_info*)D)->desc, cuidx_D.data(),
        translate_prectype(prec)))

    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(cuhandle,
                permutation_desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType)));

    assert(scalarType == translate_datatype(((cutensor_info*)D)->type));

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t plan_pref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                cuhandle,
                &plan_pref,
                algo,
                CUTENSOR_JIT_MODE_NONE));

    uint64_t workspace_size_estimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    cutensorEstimateWorkspaceSize(cuhandle,
                contraction_desc,
                plan_pref,
                workspacePref,
                &workspace_size_estimate);

    cuplan->contraction_plan = new cutensorPlan_t;
    HANDLE_ERROR(cutensorCreatePlan(cuhandle,
                cuplan->contraction_plan,
                contraction_desc,
                plan_pref,
                workspace_size_estimate));

    cuplan->permutation_plan = new cutensorPlan_t;
    HANDLE_ERROR(cutensorCreatePlan(cuhandle,
        cuplan->permutation_plan,
        permutation_desc,
        plan_pref,
        workspace_size_estimate
    ))
    cuplan->data_offset_A = ((cutensor_info*)A)->data_offset;
    cuplan->copy_size_A = ((cutensor_info*)A)->copy_size;
    cuplan->data_offset_B = ((cutensor_info*)B)->data_offset;
    cuplan->copy_size_B = ((cutensor_info*)B)->copy_size;
    cuplan->data_offset_C = ((cutensor_info*)C)->data_offset;
    cuplan->copy_size_C = ((cutensor_info*)C)->copy_size;
    cuplan->data_offset_D = ((cutensor_info*)D)->data_offset;
    cuplan->copy_size_D = ((cutensor_info*)D)->copy_size;
    cuplan->sections_D = 1;
    cuplan->section_size_D = 1;
    cuplan->sections_nmode_D = 0;
    cuplan->section_strides_D = new int64_t[TAPP_get_nmodes(D)];
    cuplan->section_extents_D = new int64_t[TAPP_get_nmodes(D)];
    cuplan->type_D = ((cutensor_info*)D)->type;
    int64_t sorted_strides_D[TAPP_get_nmodes(D)];
    memcpy(sorted_strides_D, ((cutensor_info*)D)->strides, TAPP_get_nmodes(D) * sizeof(int64_t));
    auto compare = [](int64_t a, int64_t b) { return std::abs(a) < std::abs(b); };
    std::sort(sorted_strides_D, sorted_strides_D + TAPP_get_nmodes(D), compare);
    for (int i = 0; i < TAPP_get_nmodes(D); i++)
    {
        for (int j = 0; j < TAPP_get_nmodes(D); j++)
        {
            if (((cutensor_info*)D)->strides[j] == sorted_strides_D[i])
            {
                if (std::abs(sorted_strides_D[i]) == cuplan->section_size_D)
                {
                    cuplan->section_size_D *= std::abs(((cutensor_info*)D)->extents[i]);
                }
                else
                {
                    cuplan->sections_D *= ((cutensor_info*)D)->extents[j];
                    cuplan->section_extents_D[cuplan->sections_nmode_D] = ((cutensor_info*)D)->extents[j];
                    cuplan->section_strides_D[cuplan->sections_nmode_D] = ((cutensor_info*)D)->strides[j];
                    cuplan->sections_nmode_D++;
                }
                break;
            }
        }
    }
    cuplan->section_size_D *= sizeof_datatype(((cutensor_info*)D)->type);
    *plan = (TAPP_tensor_product) cuplan;
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(contraction_desc));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(permutation_desc));
    cutensorDestroyPlanPreference(plan_pref);
    return 0; // TODO: implement cutensor error handling
}

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    cutensor_plan* cuplan = (cutensor_plan*) plan;
    HANDLE_ERROR(cutensorDestroyPlan(*cuplan->contraction_plan));
    delete cuplan->contraction_plan;
    HANDLE_ERROR(cutensorDestroyPlan(*cuplan->permutation_plan));
    delete cuplan->permutation_plan;
    delete[] cuplan->section_strides_D;
    delete[] cuplan->section_extents_D;
    delete cuplan;
    return 0; // TODO: implement cutensor error handling
}
 
TAPP_EXPORT TAPP_error TAPP_execute_product(TAPP_tensor_product plan,
                                            TAPP_executor exec,
                                            TAPP_status* status,
                                            const void* alpha,
                                            const void* A,
                                            const void* B,
                                            const void* beta,
                                            const void* C,
                                                  void* D)
{    
    void *A_d, *B_d, *C_d, *D_d, *E_d;
    cudaMalloc((void**)&A_d, ((cutensor_plan*)plan)->copy_size_A);
    cudaMalloc((void**)&B_d, ((cutensor_plan*)plan)->copy_size_B);
    cudaMalloc((void**)&C_d, ((cutensor_plan*)plan)->copy_size_C);
    cudaMalloc((void**)&D_d, ((cutensor_plan*)plan)->copy_size_D);
    cudaMalloc((void**)&E_d, ((cutensor_plan*)plan)->copy_size_D);
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, (void*)((intptr_t)A + ((cutensor_plan*)plan)->data_offset_A), ((cutensor_plan*)plan)->copy_size_A, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, (void*)((intptr_t)B + ((cutensor_plan*)plan)->data_offset_B), ((cutensor_plan*)plan)->copy_size_B, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, (void*)((intptr_t)C + ((cutensor_plan*)plan)->data_offset_C), ((cutensor_plan*)plan)->copy_size_C, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(D_d, (void*)((intptr_t)D + ((cutensor_plan*)plan)->data_offset_D), ((cutensor_plan*)plan)->copy_size_D, cudaMemcpyHostToDevice));
    A_d = (void*)((intptr_t)A_d + ((cutensor_plan*)plan)->data_offset_A);
    B_d = (void*)((intptr_t)B_d + ((cutensor_plan*)plan)->data_offset_B);
    C_d = (void*)((intptr_t)C_d + ((cutensor_plan*)plan)->data_offset_C);
    D_d = (void*)((intptr_t)D_d + ((cutensor_plan*)plan)->data_offset_D);
    E_d = (void*)((intptr_t)D_d + ((cutensor_plan*)plan)->data_offset_D);
    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);
    cutensorHandle_t handle;
    cutensorCreate(&handle);
    cutensorPlan_t* contraction_plan = ((cutensor_plan*) plan)->contraction_plan;
    uint64_t contraction_actual_workspace_size = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                *contraction_plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &contraction_actual_workspace_size,
                sizeof(contraction_actual_workspace_size)));

    void *contraction_work = nullptr;
    if (contraction_actual_workspace_size > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&contraction_work, contraction_actual_workspace_size));
        assert(uintptr_t(contraction_work) % 128 == 0);
    }

    cutensorPlan_t* permutation_plan = ((cutensor_plan*) plan)->permutation_plan;

    float one_float = 1.0f; // TODO: Needs to be adjusted to the datatype of D

    void* one_ptr = (void*)&one_float;

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    HANDLE_ERROR(cutensorContract(handle,
                *contraction_plan,
                alpha, A_d, B_d,
                beta,  C_d, D_d, 
                contraction_work, contraction_actual_workspace_size, stream));

    HANDLE_ERROR(cutensorPermute(handle,
                *permutation_plan,
                one_ptr,
                D_d,
                E_d,
                stream));

    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

    int64_t section_coordinates_D[((cutensor_plan*)plan)->sections_D];
    for (size_t i = 0; i < ((cutensor_plan*)plan)->sections_D; i++)
    {
        section_coordinates_D[i] = 0;
    }

    for (size_t i = 0; i < ((cutensor_plan*)plan)->sections_D; i++)
    {
        int64_t index = compute_index(section_coordinates_D, ((cutensor_plan*)plan)->sections_nmode_D, ((cutensor_plan*)plan)->section_strides_D);
        HANDLE_CUDA_ERROR(cudaMemcpy((void*)((intptr_t)D + index * sizeof_datatype(((cutensor_plan*)plan)->type_D)), (void*)((intptr_t)D_d + index * sizeof_datatype(((cutensor_plan*)plan)->type_D)), ((cutensor_plan*)plan)->section_size_D, cudaMemcpyDeviceToHost));
        increment_coordinates(section_coordinates_D, ((cutensor_plan*)plan)->sections_nmode_D, ((cutensor_plan*)plan)->section_extents_D);
    }

    cutensorDestroy(handle);
    cudaStreamDestroy(stream);

    A_d = (void*)((intptr_t)A_d - ((cutensor_plan*)plan)->data_offset_A);
    B_d = (void*)((intptr_t)B_d - ((cutensor_plan*)plan)->data_offset_B);
    C_d = (void*)((intptr_t)C_d - ((cutensor_plan*)plan)->data_offset_C);
    D_d = (void*)((intptr_t)D_d - ((cutensor_plan*)plan)->data_offset_D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    if (contraction_work) cudaFree(contraction_work);
    return 0; // TODO: implement cutensor error handling
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