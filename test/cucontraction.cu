#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>

#include <complex.h>

// Compile with: nvcc test/cucontraction.cu -o test/cucontraction -L/usr/lib/x86_64-linux-gnu/libcutensor/12 -I/usr/include/ -std=c++11 -lcutensor
// Run with: ./test/cucontraction

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                             \
{ const auto err = x;                                               \
    if( err != CUTENSOR_STATUS_SUCCESS )                              \
    { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                        \
    { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

int main(int argc, char** argv)
{
    // Host element type definition
    typedef std::complex<float> floatTypeA;
    typedef std::complex<float> floatTypeB;
    typedef std::complex<float> floatTypeC;
    typedef std::complex<float> floatTypeD;
    typedef std::complex<float> floatTypeCompute;

    // CUDA types
    cutensorDataType_t typeA = CUTENSOR_C_32F;
    cutensorDataType_t typeB = CUTENSOR_C_32F;
    cutensorDataType_t typeC = CUTENSOR_C_32F;
    cutensorDataType_t typeD = CUTENSOR_C_32F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

    printf("Include headers and define data types\n");

    /* ***************************** */

    // Create vector of modes
    std::vector<int> modeA{'m','v'};
    std::vector<int> modeB{'v','u'};
    std::vector<int> modeC{'m','u'};
    std::vector<int> modeD{'m','u'};
    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();
    int nmodeD = modeD.size();

    // Extents
    std::unordered_map<int, int64_t> extent;
    extent['m'] = 2;
    extent['u'] = 2;
    extent['v'] = 2;

    // Create a vector of extents for each tensor
    std::vector<int64_t> extentD;
    for(auto mode : modeD)
        extentD.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for(auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for(auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentB;
    for(auto mode : modeB)
        extentB.push_back(extent[mode]);

    printf("Define modes and extents\n");

    /* ***************************** */

    // Number of elements of each tensor
    size_t elementsA = 1;
    for(auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for(auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for(auto mode : modeC)
        elementsC *= extent[mode];
    size_t elementsD = 1;
    for(auto mode : modeD)
        elementsD *= extent[mode];

    // Size in bytes
    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeB = sizeof(floatTypeB) * elementsB;
    size_t sizeC = sizeof(floatTypeC) * elementsC;
    size_t sizeD = sizeof(floatTypeD) * elementsD;

    // Allocate on device
    void *A_d, *B_d, *C_d, *D_d;
    cudaMalloc((void**)&A_d, sizeA);
    cudaMalloc((void**)&B_d, sizeB);
    cudaMalloc((void**)&C_d, sizeC);
    cudaMalloc((void**)&D_d, sizeD);

    // Allocate on host
    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeB *B = (floatTypeB*) malloc(sizeof(floatTypeB) * elementsB);
    floatTypeC *C = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);
    floatTypeC *D = (floatTypeD*) malloc(sizeof(floatTypeD) * elementsD);

    // Initialize data on host
    for(int64_t i = 0; i < elementsA; i++)
        A[i] = {1, 1};
    for(int64_t i = 0; i < elementsB; i++)
        B[i] = {1, 1};
    for(int64_t i = 0; i < elementsC; i++)
        C[i] =  {4, 4};
    for(int64_t i = 0; i < elementsD; i++)
        D[i] =  {4, 4};

    // Copy to device
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(B_d, B, sizeB, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C, sizeC, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(D_d, D, sizeD, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);
    assert(uintptr_t(D_d) % kAlignment == 0);

    printf("Allocate, initialize and transfer tensors\n");

    /*************************
     * cuTENSOR
     *************************/ 

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descA,
                nmodeA,
                extentA.data(),
                NULL,/*stride*/
                typeA, kAlignment));

    cutensorTensorDescriptor_t descB;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descB,
                nmodeB,
                extentB.data(),
                NULL,/*stride*/
                typeB, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descC,
                nmodeC,
                extentC.data(),
                NULL,/*stride*/
                typeC, kAlignment));

    cutensorTensorDescriptor_t descD;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descD,
                nmodeD,
                extentD.data(),
                NULL,/*stride*/
                typeD, kAlignment));

    printf("Initialize cuTENSOR and tensor descriptors\n");

    /*******************************
     * Create Contraction Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateContraction(handle, 
                &desc,
                descA, modeA.data(), /* unary operator A*/CUTENSOR_OP_IDENTITY,
                descB, modeB.data(), /* unary operator B*/CUTENSOR_OP_IDENTITY,
                descC, modeC.data(), /* unary operator C*/CUTENSOR_OP_CONJ,
                descD, modeD.data(),
                descCompute));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
                desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType)));

    assert(scalarType == CUTENSOR_C_32F);
    typedef std::complex<float> floatTypeCompute;
    floatTypeCompute alpha = (floatTypeCompute){1, 0}; // If this is set to 0. The result is what I expect but not when set to anything else.
    floatTypeCompute beta  = (floatTypeCompute){1, 0};

    /**************************
     * Set the algorithm to use
     ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                handle,
                &planPref,
                algo,
                CUTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                desc,
                planPref,
                workspacePref,
                &workspaceSizeEstimate));

    /**************************
     * Create Contraction Plan
     **************************/

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                &plan,
                desc,
                planPref,
                workspaceSizeEstimate));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &actualWorkspaceSize,
                sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    /**********************
     * Execute
     **********************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    HANDLE_ERROR(cutensorContract(handle,
                plan,
                (void*) &alpha, A_d, B_d,
                (void*) &beta,  C_d, D_d, 
                work, actualWorkspaceSize, stream));

    // wait for the operation to finish
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
    printf("Contraction completed\n");
    // Copy result to host
    HANDLE_CUDA_ERROR(cudaMemcpy((void*) D, D_d, sizeC, cudaMemcpyDeviceToHost));
    printf("Result copied to host\n");
    // Print a few result entries
    for(int64_t i = 0; i < elementsC; i++)
        printf("D[%ld] = %f + %fi\n", i, D[i].real(), D[i].imag());

    /**********************
     * Free allocated data
     **********************/
    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descD));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    if (A) free(A);
    if (B) free(B);
    if (C) free(C);
    if (D) free(D);
    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    if (work) cudaFree(work);

    return 0;
}