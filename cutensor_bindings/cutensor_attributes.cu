#include "cutensor_bind.h"

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, void* value)
{
    struct handle* handle_struct = (struct handle*) attr;
    switch (key)
    {
    case 0:
        memcpy(value, (void*)handle_struct->attributes[0], sizeof(bool));
        break;
    
    default:
        return 15; // Invalid key
    }
    return 0;
}

TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, void** value)
{
    struct handle* handle_struct = (struct handle*) attr;
    switch (key)
    {
    case 0:
        memcpy((void*)handle_struct->attributes[0], value, sizeof(bool));
        break;
    
    default:
        return 15; // Invalid key
    }
    return 0;
}

TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_key key)
{
    struct handle* handle_struct = (struct handle*) attr;
    switch (key)
    {
    case 0:
        {
            bool default_value = false;
            memcpy((void*)handle_struct->attributes[0], &default_value, sizeof(bool));
        }
        break;
    
    default:
        return 15; // Invalid key
    }
    return 0;
}