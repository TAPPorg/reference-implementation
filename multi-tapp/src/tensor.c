#include "../include/multi_tapp/tensor.h"
#include "internal.h"

TAPP_error Multi_TAPP_create_tensor_info(Multi_TAPP_backend backend,
                                         TAPP_tensor_info* info,
                                         TAPP_datatype type,
                                         int nmode,
                                         const int64_t* extents,
                                         const int64_t* strides)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (info == NULL)
    {
        fprintf(stderr, "ERROR: Called function with NULL info pointer");
        return MULTI_TAPP_ERROR_INVALID_ARGUMENT;
    }

    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_create_tensor_info == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_tensor_info\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = malloc(sizeof(struct Multi_TAPP_tensor_info));
    if (multi_tapp_info == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate space for a tensor info object\n");
        return MULTI_TAPP_ERROR_ALLOCATION_FAILED;
    }
    multi_tapp_info->backend = backend;

    TAPP_error error = backend_ptr->TAPP_create_tensor_info(&multi_tapp_info->info, type, nmode, extents, strides);
    
    if (!multi_tapp_check_success(error))
    {
        free(multi_tapp_info);
        return multi_tapp_error_from_backend(error);
    }

    *info = (TAPP_tensor_info)multi_tapp_info;
    return MULTI_TAPP_SUCCESS;
}

TAPP_error Multi_TAPP_destroy_tensor_info(Multi_TAPP_backend backend,
                                          TAPP_tensor_info info)
{
    if (info == 0)
    {
        fprintf(stderr, "WARNING: Tried to destroy exec 0\n");
        return MULTI_TAPP_SUCCESS;
    }
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_destroy_tensor_info == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_tensor_info\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_destroy_tensor_info(multi_tapp_info->info);
    free(multi_tapp_info);

    return multi_tapp_error_from_backend(error);
}

int Multi_TAPP_get_nmodes(Multi_TAPP_backend backend,
                          TAPP_tensor_info info)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return -1;
    }
    if (info == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info\n");
        return -1;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_get_nmodes == NULL)
    {
        // Current interface does not return an error
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_nmodes\n");
        //return MULTI_TAPP_ERROR_UNIMPLEMENTED;
        return -1;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND;
        return -1;
    }

    return backend_ptr->TAPP_get_nmodes(multi_tapp_info->info);
}

TAPP_error Multi_TAPP_set_nmodes(Multi_TAPP_backend backend,
                                 TAPP_tensor_info info,
                                 int nmodes)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (info == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_INFO;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_set_nmodes == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_nmodes\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_set_nmodes(multi_tapp_info->info, nmodes);

    return multi_tapp_error_from_backend(error);
}

void Multi_TAPP_get_extents(Multi_TAPP_backend backend,
                            TAPP_tensor_info info,
                            int64_t* extents)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return;
    }
    if (info == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info\n");
        return;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_get_extents == NULL)
    {
        // Current interface does not return an error
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_extents\n");
        //return MULTI_TAPP_ERROR_UNIMPLEMENTED;
        return;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND;
        return;
    }

    backend_ptr->TAPP_get_extents(multi_tapp_info->info, extents);
}

TAPP_error Multi_TAPP_set_extents(Multi_TAPP_backend backend,
                                  TAPP_tensor_info info,
                                  const int64_t* extents)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (info == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_INFO;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_set_extents == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_extents\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_set_extents(multi_tapp_info->info, extents);

    return multi_tapp_error_from_backend(error);
}

void Multi_TAPP_get_strides(Multi_TAPP_backend backend,
                            TAPP_tensor_info info,
                            int64_t* strides)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return;
    }
    if (info == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info\n");
        return;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_get_strides == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_strides\n");
        // Current interface does not return an error
        //return MULTI_TAPP_ERROR_UNIMPLEMENTED;
        return;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND;
        return;
    }

    backend_ptr->TAPP_get_strides(multi_tapp_info->info, strides);
}

TAPP_error Multi_TAPP_set_strides(Multi_TAPP_backend backend,
                                  TAPP_tensor_info info,
                                  const int64_t* strides)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (info == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL tensor info\n");
        return MULTI_TAPP_ERROR_INVALID_TENSOR_INFO;
    }
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_set_strides == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_strides\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_set_strides(multi_tapp_info->info, strides);

    return multi_tapp_error_from_backend(error);
}
