#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>

#include "../src/tapp.h"

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

cutensorDataType_t translate_datatype(TAPP_datatype type);

cutensorComputeDescriptor_t translate_prectype(TAPP_prectype prec);

cutensorOperator_t translate_operator(TAPP_element_op op);

//TAPP_handle create_TAPP_handle();

TAPP_EXPORT TAPP_error create_handle(TAPP_handle* handle);

TAPP_EXPORT TAPP_error create_executor(TAPP_executor* exec);

typedef struct 
{
    int nmode;
    int64_t *extents;
    int64_t *strides;
    size_t elements;
    size_t size;
    cutensorTensorDescriptor_t* desc;
} cutensor_info;

typedef struct 
{
    size_t sizeA;
    size_t sizeB;
    size_t sizeC;
    size_t sizeD;
    cutensorPlan_t* plan;
} cutensor_plan;