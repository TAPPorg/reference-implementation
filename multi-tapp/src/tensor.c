#include "../include/tensor.h"

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_handle handle,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_create_tensor_info == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_tensor_info\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = malloc(sizeof(struct Multi_TAPP_tensor_info));
    multi_tapp_info->impl_id = multi_tapp_handle->impl_id;
    *info = (TAPP_tensor_info)multi_tapp_info;

    return multi_tapp_handle->TAPP_create_tensor_info(multi_tapp_info->info, *multi_tapp_handle->tapp_handle, type, nmode, extents, strides);
}

TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info,
                                    TAPP_handle handle)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_destroy_tensor_info == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_tensor_info\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (multi_tapp_handle->impl_id != multi_tapp_info->impl_id)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return 7; // TODO: Return error for incompatible tensor_info
    }

    TAPP_error error = multi_tapp_handle->TAPP_destroy_tensor_info(*multi_tapp_info->info, *multi_tapp_handle->tapp_handle);
    free(multi_tapp_info);

    return error;
}

int TAPP_get_nmodes(TAPP_tensor_info info,
                    TAPP_handle handle)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_get_nmodes == NULL)
    {
        // Current interface does not return an error
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_nmodes\n");
        //return 6; // TODO: Return error for non implemented function
        return -1;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (multi_tapp_handle->impl_id != multi_tapp_info->impl_id)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return 7; // TODO: Return error for incompatible tensor_info
        return -1;
    }

    return multi_tapp_handle->TAPP_get_nmodes(*multi_tapp_info->info, *multi_tapp_handle->tapp_handle);
}

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           TAPP_handle handle,
                           int nmodes)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_set_nmodes == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_nmodes\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (multi_tapp_handle->impl_id != multi_tapp_info->impl_id)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return 7; // TODO: Return error for incompatible tensor_info
    }

    return multi_tapp_handle->TAPP_set_nmodes(*multi_tapp_info->info, *multi_tapp_handle->tapp_handle, nmodes);
}

void TAPP_get_extents(TAPP_tensor_info info,
                      TAPP_handle handle,
                      int64_t* extents)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_get_extents == NULL)
    {
        // Current interface does not return an error
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_extents\n");
        //return 6; // TODO: Return error for non implemented function
        return;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (multi_tapp_handle->impl_id != multi_tapp_info->impl_id)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return 7; // TODO: Return error for incompatible tensor_info
        return;
    }

    return multi_tapp_handle->TAPP_get_extents(*multi_tapp_info->info, *multi_tapp_handle->tapp_handle, extents);
}

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            TAPP_handle handle,
                            const int64_t* extents)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_set_extents == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_extents\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (multi_tapp_handle->impl_id != multi_tapp_info->impl_id)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return 7; // TODO: Return error for incompatible tensor_info
    }

    return multi_tapp_handle->TAPP_set_extents(*multi_tapp_info->info, *multi_tapp_handle->tapp_handle, extents);
}

void TAPP_get_strides(TAPP_tensor_info info,
                      TAPP_handle handle,
                      int64_t* strides)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_get_strides == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_strides\n");
        // Current interface does not return an error
        //return 6; // TODO: Return error for non implemented function
        return;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (multi_tapp_handle->impl_id != multi_tapp_info->impl_id)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return 7; // TODO: Return error for incompatible tensor_info
        return;
    }

    return multi_tapp_handle->TAPP_get_strides(*multi_tapp_info->info, *multi_tapp_handle->tapp_handle, strides);
}

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            TAPP_handle handle,
                            const int64_t* strides)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_set_strides == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_strides\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (multi_tapp_handle->impl_id != multi_tapp_info->impl_id)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return 7; // TODO: Return error for incompatible tensor_info
    }

    return multi_tapp_handle->TAPP_set_strides(*multi_tapp_info->info, *multi_tapp_handle->tapp_handle, strides);
}