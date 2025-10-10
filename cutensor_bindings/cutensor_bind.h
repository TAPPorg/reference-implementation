#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>
#include <complex>

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

TAPP_EXPORT TAPP_error create_handle(TAPP_handle* handle);

TAPP_EXPORT TAPP_error create_executor(TAPP_executor* exec);

size_t sizeof_datatype(TAPP_datatype type);

typedef struct 
{
    int nmode;
    int64_t *extents;
    int64_t *strides;
    size_t elements;
    size_t copy_size;
    int64_t data_offset;
    TAPP_datatype type;
    cutensorTensorDescriptor_t* desc;
} cutensor_info;

typedef struct 
{
    int64_t data_offset_A;
    size_t copy_size_A;
    int64_t data_offset_B;
    size_t copy_size_B;
    int64_t data_offset_C;
    size_t copy_size_C;
    int64_t data_offset_D;
    size_t copy_size_D;
    int64_t sections_D;
    int64_t section_size_D;
    int64_t sections_nmode_D;
    int64_t* section_extents_D;
    int64_t* section_strides_D;
    TAPP_datatype type_D;
    cutensorPlan_t* plan;
} cutensor_plan;