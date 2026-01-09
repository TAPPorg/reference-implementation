#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>
#include <complex>
#include <cstdint> // uint64_t

#include "../src/tapp.h"

#define ATTR_KEY_USE_DEVICE_MEMORY 0

cutensorDataType_t translate_datatype(TAPP_datatype type);

cutensorComputeDescriptor_t translate_prectype(TAPP_prectype prec, TAPP_datatype datatype);

cutensorOperator_t translate_operator(TAPP_element_op op);

TAPP_EXPORT TAPP_error create_handle(TAPP_handle* handle);

TAPP_EXPORT TAPP_error create_executor(TAPP_executor* exec);

size_t sizeof_datatype(TAPP_datatype type);

int pack_error(int current_value, int tapp_err);
int pack_error(int current_value, cutensorStatus_t e); 
int pack_error(int current_value, cudaError_t e);

struct handle
{
    cutensorHandle_t* libhandle;
    intptr_t* attributes;
};

struct tensor_info
{
    int nmode;
    int64_t *extents;
    int64_t *strides;
    size_t elements;
    size_t copy_size;
    int64_t data_offset;
    TAPP_datatype type;
    cutensorTensorDescriptor_t* desc;
};

struct product_plan
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
    cutensorPlan_t* contraction_plan;
    cutensorPlan_t* permutation_plan;
    cutensorHandle_t* handle;
};
