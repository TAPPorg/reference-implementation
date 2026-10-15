#include "../include/multi_tapp/attributes.h"
#include "internal.h"

TAPP_error Multi_TAPP_create_attr(Multi_TAPP_backend backend, TAPP_attr *attr)
{
    if (attr == NULL)
    {
        fprintf(stderr, "ERROR: Called function with NULL attr pointer");
        return MULTI_TAPP_ERROR_INVALID_ARGUMENT;
    }

    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_create_attr == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_create_attr\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_attr* multi_tapp_attr = malloc(sizeof(struct mt_attr));
    if (multi_tapp_attr == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate space for an attribute object\n");
        return MULTI_TAPP_ERROR_ALLOCATION_FAILED;
    }
    multi_tapp_attr->backend = backend;

    TAPP_error error = backend_ptr->TAPP_create_attr(&multi_tapp_attr->attr);

    if (!multi_tapp_check_success(error))
    {
        free(multi_tapp_attr);
        return multi_tapp_error_from_backend(error);
    }

    *attr = (TAPP_attr)multi_tapp_attr;
    return MULTI_TAPP_SUCCESS;
}

TAPP_error Multi_TAPP_destroy_attr(Multi_TAPP_backend backend, TAPP_attr attr)
{
    if (attr == 0)
    {
        fprintf(stderr, "WARNING: Tried to destroy attr 0\n");
        return MULTI_TAPP_SUCCESS;
    }
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_destroy_attr == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_destroy_attr\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_attr* multi_tapp_attr = (struct mt_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return MULTI_TAPP_ERROR_ATTR_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_destroy_attr(multi_tapp_attr->attr);
    free(multi_tapp_attr);

    return multi_tapp_error_from_backend(error);
}

TAPP_error Multi_TAPP_attr_set(Multi_TAPP_backend backend, TAPP_attr attr, TAPP_key key, size_t size, void* value)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (attr == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL attribute\n");
        return MULTI_TAPP_ERROR_INVALID_ATTR;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_attr_set == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_set\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_attr* multi_tapp_attr = (struct mt_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return MULTI_TAPP_ERROR_ATTR_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_attr_set(multi_tapp_attr->attr, key, size, value);

    return multi_tapp_error_from_backend(error);
}

TAPP_error Multi_TAPP_attr_get(Multi_TAPP_backend backend, TAPP_attr attr, TAPP_key key, size_t *size, void** value)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (attr == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL attribute\n");
        return MULTI_TAPP_ERROR_INVALID_ATTR;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_attr_get == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_get\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_attr* multi_tapp_attr = (struct mt_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return MULTI_TAPP_ERROR_ATTR_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_attr_get(multi_tapp_attr->attr, key, size, value);

    return multi_tapp_error_from_backend(error);
}

TAPP_error Multi_TAPP_attr_clear(Multi_TAPP_backend backend, TAPP_attr attr, TAPP_key key)
{
    if (backend == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL backend\n");
        return MULTI_TAPP_ERROR_INVALID_BACKEND;
    }
    if (attr == 0)
    {
        fprintf(stderr, "ERROR: Called function with NULL attribute\n");
        return MULTI_TAPP_ERROR_INVALID_ATTR;
    }
    struct mt_backend* backend_ptr = (struct mt_backend*)backend;
    if (backend_ptr->TAPP_attr_clear == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_clear\n");
        return MULTI_TAPP_ERROR_UNIMPLEMENTED;
    }

    struct mt_attr* multi_tapp_attr = (struct mt_attr*)attr;
    if (backend != multi_tapp_attr->backend)
    {
        return MULTI_TAPP_ERROR_ATTR_WRONG_BACKEND;
    }

    TAPP_error error = backend_ptr->TAPP_attr_clear(multi_tapp_attr->attr, key);

    return multi_tapp_error_from_backend(error);
}