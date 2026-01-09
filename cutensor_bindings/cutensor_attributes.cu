#include "cutensor_bind.h"
#include "../src/tapp/handle.h"
#include "../src/tapp/attributes.h"

TAPP_EXPORT TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, void* value)
{
    struct handle* handle_struct = (struct handle*) attr;
    switch (key)
    {
    case 0:
        memcpy(value, (void*)handle_struct->attributes[0], sizeof(bool));
        break;
    
    default:
        // Invalid key
        break;
    }
    return 0; // TODO: implement cutensor error handling
}

TAPP_EXPORT TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, void** value)
{
    struct handle* handle_struct = (struct handle*) attr;
    switch (key)
    {
    case 0:
        memcpy((void*)handle_struct->attributes[0], value, sizeof(bool));
        break;
    
    default:
        // Invalid key
        break;
    }
    return 0; // TODO: implement cutensor error handling
}

TAPP_EXPORT TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_key key)
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
        // Invalid key
        break;
    }
    return 0; // TODO: implement cutensor error handling
}