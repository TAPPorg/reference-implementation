#include "../include/tensor.h"

TAPP_error Multi_TAPP_create_tensor_info(TAPP_backend backend,
                                         TAPP_tensor_info* info,
                                         TAPP_datatype type,
                                         int nmode,
                                         const int64_t* extents,
                                         const int64_t* strides)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_create_tensor_info == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_tensor_info\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = malloc(sizeof(struct Multi_TAPP_tensor_info));
    multi_tapp_info->backend = backend;
    *info = (TAPP_tensor_info)multi_tapp_info;

    return backend_ptr->TAPP_create_tensor_info(&multi_tapp_info->info, type, nmode, extents, strides);
}

TAPP_error Multi_TAPP_destroy_tensor_info(TAPP_backend backend,
                                          TAPP_tensor_info info)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_destroy_tensor_info == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_tensor_info\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return 7; // TODO: Return error for incompatible tensor_info
    }

    TAPP_error error = backend_ptr->TAPP_destroy_tensor_info(multi_tapp_info->info);
    free(multi_tapp_info);

    return error;
}

int Multi_TAPP_get_nmodes(TAPP_backend backend,
                          TAPP_tensor_info info)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_get_nmodes == NULL)
    {
        // Current interface does not return an error
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_nmodes\n");
        //return 6; // TODO: Return error for non implemented function
        return -1;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return 7; // TODO: Return error for incompatible tensor_info
        return -1;
    }

    return backend_ptr->TAPP_get_nmodes(multi_tapp_info->info);
}

TAPP_error Multi_TAPP_set_nmodes(TAPP_backend backend,
                                 TAPP_tensor_info info,
                                 int nmodes)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_set_nmodes == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_nmodes\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return 7; // TODO: Return error for incompatible tensor_info
    }

    return backend_ptr->TAPP_set_nmodes(multi_tapp_info->info, nmodes);
}

void Multi_TAPP_get_extents(TAPP_backend backend,
                            TAPP_tensor_info info,
                            int64_t* extents)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_get_extents == NULL)
    {
        // Current interface does not return an error
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_extents\n");
        //return 6; // TODO: Return error for non implemented function
        return;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return 7; // TODO: Return error for incompatible tensor_info
        return;
    }

    backend_ptr->TAPP_get_extents(multi_tapp_info->info, extents);
}

TAPP_error Multi_TAPP_set_extents(TAPP_backend backend,
                                  TAPP_tensor_info info,
                                  const int64_t* extents)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_set_extents == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_extents\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return 7; // TODO: Return error for incompatible tensor_info
    }

    return backend_ptr->TAPP_set_extents(multi_tapp_info->info, extents);
}

void Multi_TAPP_get_strides(TAPP_backend backend,
                            TAPP_tensor_info info,
                            int64_t* strides)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_get_strides == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_get_strides\n");
        // Current interface does not return an error
        //return 6; // TODO: Return error for non implemented function
        return;
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        // Current interface does not return an error
        //return 7; // TODO: Return error for incompatible tensor_info
        return;
    }

    backend_ptr->TAPP_get_strides(multi_tapp_info->info, strides);
}

TAPP_error Multi_TAPP_set_strides(TAPP_backend backend,
                                  TAPP_tensor_info info,
                                  const int64_t* strides)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_set_strides == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_set_strides\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_tensor_info* multi_tapp_info = (struct Multi_TAPP_tensor_info*)info;
    if (backend != multi_tapp_info->backend)
    {
        fprintf(stderr, "ERROR: TAPP_tensor_info from other implementation\n");
        return 7; // TODO: Return error for incompatible tensor_info
    }

    return backend_ptr->TAPP_set_strides(multi_tapp_info->info, strides);
}
