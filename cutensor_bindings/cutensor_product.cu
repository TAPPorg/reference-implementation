#include "../src/tapp/product.h"
#include "cutensor_bind.h"

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
    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(cuhandle, 
                &desc,
                *((cutensor_info*)A)->desc, cuidx_A.data(), translate_operator(op_A),
                *((cutensor_info*)B)->desc, cuidx_B.data(), translate_operator(op_B),
                *((cutensor_info*)C)->desc, cuidx_C.data(), translate_operator(op_C),
                *((cutensor_info*)D)->desc, cuidx_D.data(),
                translate_prectype(prec)));

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(cuhandle,
                desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                cuhandle,
                &planPref,
                algo,
                CUTENSOR_JIT_MODE_NONE));

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    cutensorEstimateWorkspaceSize(cuhandle,
                desc,
                planPref,
                workspacePref,
                &workspaceSizeEstimate);

    cuplan->plan = new cutensorPlan_t;
    HANDLE_ERROR(cutensorCreatePlan(cuhandle,
                cuplan->plan,
                desc,
                planPref,
                workspaceSizeEstimate));
    cuplan->sizeA = ((cutensor_info*)A)->size;
    cuplan->sizeB = ((cutensor_info*)B)->size;
    cuplan->sizeC = ((cutensor_info*)C)->size;
    cuplan->sizeD = ((cutensor_info*)D)->size;
    *plan = (TAPP_tensor_product) cuplan;
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    cutensorDestroyPlanPreference(planPref);
    return 0; // TODO: implement cutensor error handling
}

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan)
{
    cutensor_plan* cuplan = (cutensor_plan*) plan;
    HANDLE_ERROR(cutensorDestroyPlan(*cuplan->plan));
    delete cuplan->plan;
    delete cuplan;
    return 0; // TODO: implement cutensor error handling
}
 
//TODO: in-place operation: set C = NULL or TAPP_IN_PLACE?
 
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
    void *A_d, *B_d, *C_d, *D_d;
    cudaMalloc((void**)&A_d, ((cutensor_plan*)plan)->sizeA);
    cudaMalloc((void**)&B_d, ((cutensor_plan*)plan)->sizeB);
    cudaMalloc((void**)&C_d, ((cutensor_plan*)plan)->sizeC);
    cudaMalloc((void**)&D_d, ((cutensor_plan*)plan)->sizeD);
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, ((cutensor_plan*)plan)->sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, ((cutensor_plan*)plan)->sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, ((cutensor_plan*)plan)->sizeC, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(D_d, D, ((cutensor_plan*)plan)->sizeD, cudaMemcpyHostToDevice));
    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);
    cutensorHandle_t handle;
    cutensorCreate(&handle);
    cutensorPlan_t* cuplan = ((cutensor_plan*) plan)->plan;
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                *cuplan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &actualWorkspaceSize,
                sizeof(actualWorkspaceSize)));

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0);
    }
    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    HANDLE_ERROR(cutensorContract(handle,
                *cuplan,
                alpha, A_d, B_d,
                beta,  C_d, D_d, 
                work, actualWorkspaceSize, stream));

    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
    HANDLE_CUDA_ERROR(cudaMemcpy((void*) D, D_d, ((cutensor_plan*)plan)->sizeD, cudaMemcpyDeviceToHost));

    cutensorDestroy(handle);
    cudaStreamDestroy(stream);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    if (work) cudaFree(work);
    return 0; // TODO: implement cutensor error handling
}