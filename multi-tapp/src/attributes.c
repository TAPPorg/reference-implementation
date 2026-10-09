#include "../include/attributes.h"

TAPP_error Multi_TAPP_create_attr(TAPP_backend backend, TAPP_attr *attr)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_create_attr == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_attr\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_attr* multi_tapp_attr = malloc(sizeof(struct Multi_TAPP_attr));
    multi_tapp_attr->backend = backend;
    *attr = (TAPP_executor)multi_tapp_attr;

    return backend_ptr->TAPP_create_attr(&multi_tapp_attr->attr);
}

TAPP_error Multi_TAPP_destroy_attr(TAPP_backend backend, TAPP_attr attr)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_destroy_attr == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_attr\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_attr* multi_tapp_attr = (struct Multi_TAPP_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    TAPP_error error = backend_ptr->TAPP_destroy_executor(multi_tapp_attr->attr);
    free(multi_tapp_attr);

    return error;
}

TAPP_error Multi_TAPP_attr_set(TAPP_backend backend, TAPP_attr attr, TAPP_key key, size_t size, void* value)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_attr_set == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_set\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_attr* multi_tapp_attr = (struct Multi_TAPP_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    TAPP_error error = backend_ptr->TAPP_attr_set(multi_tapp_attr->attr, key, size, value);

    return error;
}

TAPP_error Multi_TAPP_attr_get(TAPP_backend backend, TAPP_attr attr, TAPP_key key, size_t *size, void** value)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_attr_get == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_get\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_attr* multi_tapp_attr = (struct Multi_TAPP_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    TAPP_error error = backend_ptr->TAPP_attr_get(multi_tapp_attr->attr, key, size, value);

    return error;
}

TAPP_error Multi_TAPP_attr_clear(TAPP_backend backend, TAPP_attr attr, TAPP_key key)
{
    struct Multi_TAPP_backend* backend_ptr = (struct Multi_TAPP_backend*)backend;
    if (backend_ptr->TAPP_attr_clear == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_get\n");
        return 6; // TODO: Return error for non implemented function
    }

    struct Multi_TAPP_attr* multi_tapp_attr = (struct Multi_TAPP_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return 8; // TODO: Return error for incompatible handle
    }

    TAPP_error error = backend_ptr->TAPP_attr_clear(multi_tapp_attr->attr, key);

    return error;
}