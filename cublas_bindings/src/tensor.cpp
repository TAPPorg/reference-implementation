#include "../include/tensor.h"

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_handle handle,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides)
{
    (void)handle;
    struct tensor_info* tensor_info = new struct tensor_info;

    size_t elements = 1;
    for (int i = 0; i < nmode; ++i)
        elements *= extents[i];

    // copy_size / data_offset bound the memory span of the (possibly
    // negative-strided) tensor, matching the cuTENSOR bindings so host<->device
    // transfers move exactly the referenced bytes.
    tensor_info->copy_size = 1;
    tensor_info->data_offset = 0;
    for (int i = 0; i < nmode; i++)
    {
        tensor_info->copy_size += (extents[i] - 1) * strides[i];
        if (strides[i] < 0)
        {
            tensor_info->data_offset += extents[i] * strides[i];
        }
    }
    tensor_info->copy_size *= sizeof_datatype(type);
    tensor_info->data_offset *= sizeof_datatype(type);
    tensor_info->type = type;
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
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info)
{
    struct tensor_info* tensor_info = (struct tensor_info*) info;
    delete[] tensor_info->extents;
    delete[] tensor_info->strides;
    delete tensor_info;
    return TAPP_SUCCESS;
}

int TAPP_get_nmodes(TAPP_tensor_info info)
{
    return ((struct tensor_info*) info)->nmode;
}

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           int nmodes)
{
    (void)info;
    (void)nmodes;
    return tapp_error(TAPP_ERROR_TYPE_TAPP, TAPP_ERR_NOT_IMPLEMENTED); // Not supported: would require reallocating extents/strides.
}

void TAPP_get_extents(TAPP_tensor_info info,
                      int64_t* extents)
{
    memcpy(extents, ((struct tensor_info*) info)->extents, ((struct tensor_info*) info)->nmode * sizeof(int64_t));
}

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            const int64_t* extents)
{
    (void)info;
    (void)extents;
    return tapp_error(TAPP_ERROR_TYPE_TAPP, TAPP_ERR_NOT_IMPLEMENTED); // Not supported.
}

void TAPP_get_strides(TAPP_tensor_info info,
                      int64_t* strides)
{
    memcpy(strides, ((struct tensor_info*) info)->strides, ((struct tensor_info*) info)->nmode * sizeof(int64_t));
}

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            const int64_t* strides)
{
    (void)info;
    (void)strides;
    return tapp_error(TAPP_ERROR_TYPE_TAPP, TAPP_ERR_NOT_IMPLEMENTED); // Not supported.
}
