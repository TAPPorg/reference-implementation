#include "../include/attributes.h"

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, void* value)
{
    struct handle* handle_struct = (struct handle*) attr;
    switch (key)
    {
    case ATTR_KEY_USE_DEVICE_MEMORY:
        memcpy((void*)handle_struct->attributes[0], value, sizeof(bool));
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
    case ATTR_KEY_USE_DEVICE_MEMORY:
        memcpy(value, (void*)handle_struct->attributes[0], sizeof(bool));
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
    case ATTR_KEY_USE_DEVICE_MEMORY:
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
