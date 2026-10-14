#include "../include/attributes.h"
#include "../include/executor.h"

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, void* value)
{
    /*
    struct executor* executor_struct = (struct executor*) attr;
    switch (key)
    {
    case 0:
        memcpy((void*)executor_struct->attributes[0], value, sizeof(bool));
        break;
    
    default:
        return TAPP_ERR_INVALID_KEY;
    } */
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, void** value)
{
    /*
    struct executor* executor_struct = (struct executor*) attr;
    switch (key)
    {
    case 0:
        *value = (void*)executor_struct->attributes[0];
        break;
    
    default:
        return TAPP_ERR_INVALID_KEY;
    } */
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_key key)
{
    /*
    struct executor* executor_struct = (struct executor*) attr;
    switch (key)
    {
    case 0:
        {
            bool default_value = false;
            memcpy((void*)executor_struct->attributes[0], &default_value, sizeof(bool));
        }
        break;
    
    default:
        return TAPP_ERR_INVALID_KEY;
    } */
    return TAPP_SUCCESS;
}
