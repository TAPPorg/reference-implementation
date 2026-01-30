#include "../include/attributes.h"

/*
 * Not sure how this will work reliably.
 */

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_handle handle, TAPP_key key, void* value)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_attr_set == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_set\n");
        return 6; // TODO: Return error for non implemented function
    }

    return multi_tapp_handle->TAPP_attr_set(attr, *multi_tapp_handle->tapp_handle, key, value);
}

TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_handle handle, TAPP_key key, void** value)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_attr_get == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_get\n");
        return 6; // TODO: Return error for non implemented function
    }

    return multi_tapp_handle->TAPP_attr_get(attr, *multi_tapp_handle->tapp_handle, key, value);
}

TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_handle handle, TAPP_key key)
{
    struct Multi_TAPP_handle* multi_tapp_handle = (struct Multi_TAPP_handle*)handle;
    if (multi_tapp_handle->TAPP_attr_clear == NULL)
    {
        fprintf(stderr, "ERROR: Called unimplemented function TAPP_attr_clear\n");
        return 6; // TODO: Return error for non implemented function
    }

    return multi_tapp_handle->TAPP_attr_clear(attr, *multi_tapp_handle->tapp_handle, key);
}