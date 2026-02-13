#include "../include/tensor.h"

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_handle handle,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides)
{
    struct tensor_info* tensor_info = new struct tensor_info;
    tensor_info->desc = new cutensorTensorDescriptor_t;
    struct handle* handle_struct = (struct handle*) handle;
    
    const uint32_t kAlignment = 128;
    cutensorStatus_t err = cutensorCreateTensorDescriptor(*handle_struct->libhandle,
                tensor_info->desc,
                nmode,
                extents,
                strides,
                translate_datatype(type), kAlignment);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        delete tensor_info->desc;
        delete tensor_info;
        return pack_error(0, err);
    }
    size_t elements = 1;
    for (int i = 0; i < nmode; ++i)
        elements *= extents[i];
    tensor_info->copy_size = 1;
    tensor_info->data_offset = 0;
    for (int i = 0; i < nmode; i++)
    {
        tensor_info->copy_size += (extents[i] - 1)*strides[i];
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
    return 0;
}

TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info)
{
    struct tensor_info* tensor_info = (struct tensor_info*) info;
    cutensorStatus_t err = cutensorDestroyTensorDescriptor(*tensor_info->desc);
    if (err != CUTENSOR_STATUS_SUCCESS)
    {
        return pack_error(0, err);
    }
    delete tensor_info->desc;
    delete[] tensor_info->extents;
    delete[] tensor_info->strides;
    delete tensor_info;
    return 0;
}

int TAPP_get_nmodes(TAPP_tensor_info info)
{
    return ((struct tensor_info*) info)->nmode;
}

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           int nmodes)
{
    return -1; // Can for now not be implemented. Cutensor does not support changing the number of modes after creation, so this would require recreating the descriptor, would need handle.
}

void TAPP_get_extents(TAPP_tensor_info info,
                      int64_t* extents)
{
    memcpy(extents, ((struct tensor_info*) info)->extents, ((struct tensor_info*) info)->nmode * sizeof(int64_t));
    return; 
}

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            const int64_t* extents)
{
    return -1; // Can for now not be implemented. Cutensor does not support changing the number of modes after creation, so this would require recreating the descriptor, would need handle.
}

void TAPP_get_strides(TAPP_tensor_info info,
                      int64_t* strides)
{
    memcpy(strides, ((struct tensor_info*) info)->strides, ((struct tensor_info*) info)->nmode * sizeof(int64_t));
    return; 
}

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            const int64_t* strides)
{
    return -1; // Can for now not be implemented. Cutensor does not support changing the number of modes after creation, so this would require recreating the descriptor, would need handle.
}