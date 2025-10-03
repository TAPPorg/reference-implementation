#include "../src/tapp/tensor.h"
#include "cutensor_bind.h"

TAPP_EXPORT TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                               TAPP_datatype type,
                                               int nmode,
                                               const int64_t* extents,
                                               const int64_t* strides)
{
    cutensorHandle_t handle;
    cutensorCreate(&handle);
    cutensor_info* tensor_info = new cutensor_info;
    tensor_info->desc = new cutensorTensorDescriptor_t;
    const uint32_t kAlignment = 128;
    cutensorCreateTensorDescriptor(handle,
                tensor_info->desc,
                nmode,
                extents,
                strides,
                translate_datatype(type), kAlignment);
    cutensorDestroy(handle);
    size_t elements = 1;
    for (int i = 0; i < nmode; ++i)
        elements *= extents[i];
    size_t size = elements;
    switch (translate_datatype(type))
    {
    case CUTENSOR_R_32F:
        size *= sizeof(float);
        break;
    case CUTENSOR_R_64F:
        size *= sizeof(double);
        break;
    /*case CUTENSOR_C_32F: //TODO: Fix these types
        size *= sizeof(complex float);
        break;
    case CUTENSOR_C_64F:
        size *= sizeof(complex double);
        break;
    case CUTENSOR_R_16F:
        size *= sizeof(__half);
        break;
    case CUTENSOR_R_16BF:
        size *= sizeof(__nv_bfloat16);
        break;
    */
    default: // TODO: Default should probably be an error
        size *= sizeof(float); 
        break;
    }
    tensor_info->size = size;
    tensor_info->elements = elements;
    tensor_info->nmode = nmode;
    tensor_info->extents = new int64_t[nmode];
    tensor_info->strides = new int64_t[nmode];
    for (int i = 0; i < nmode; ++i)
    {
        tensor_info->extents[i] = extents[i];
        tensor_info->strides[i] = strides[i];
    }
    *info = (TAPP_tensor_info) tensor_info;
    return 0; // TODO: implement cutensor error handling
}

TAPP_EXPORT TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info)
{
    cutensor_info* tensor_info = (cutensor_info*) info;
    cutensorDestroyTensorDescriptor(*tensor_info->desc);
    delete tensor_info->desc;
    delete[] tensor_info->extents;
    delete[] tensor_info->strides;
    delete tensor_info;
    return 0; // TODO: implement cutensor error handling
}

TAPP_EXPORT int TAPP_get_nmodes(TAPP_tensor_info info)
{
    return ((cutensor_info*) info)->nmode;
}

TAPP_EXPORT TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                                       int nmodes)
{
    return 0; // TODO: correctly implement, currently placeholder
}

TAPP_EXPORT void TAPP_get_extents(TAPP_tensor_info info,
                                  int64_t* extents)
{
    memcpy(extents, ((cutensor_info*) info)->extents, ((cutensor_info*) info)->nmode * sizeof(int64_t));
    return; // TODO: correctly implement, currently placeholder
}

TAPP_EXPORT TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                                        const int64_t* extents)
{
    return 0; // TODO: correctly implement, currently placeholder
}

TAPP_EXPORT void TAPP_get_strides(TAPP_tensor_info info,
                                  int64_t* strides)
{
    memcpy(strides, ((cutensor_info*) info)->strides, ((cutensor_info*) info)->nmode * sizeof(int64_t));
    return; // TODO: correctly implement, currently placeholder
}

TAPP_EXPORT TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                                        const int64_t* strides)
{
    return 0; // TODO: correctly implement, currently placeholder
}