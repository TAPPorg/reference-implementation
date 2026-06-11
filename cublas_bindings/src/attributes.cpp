#include "../include/attributes.h"

// Number of bytes stored for each attribute key.
static size_t attr_size(TAPP_key key)
{
    switch (key)
    {
    case ATTR_KEY_USE_DEVICE_MEMORY: return sizeof(bool);
    case ATTR_KEY_PRECISION_DIGITS:  return sizeof(int);
    default:                         return 0;
    }
}

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, void* value)
{
    struct handle* handle_struct = (struct handle*) attr;
    size_t size = attr_size(key);
    if (size == 0) return 15; // Invalid key
    memcpy((void*)handle_struct->attributes[key], value, size);
    return 0;
}

TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, void** value)
{
    struct handle* handle_struct = (struct handle*) attr;
    size_t size = attr_size(key);
    if (size == 0) return 15; // Invalid key
    memcpy(value, (void*)handle_struct->attributes[key], size);
    return 0;
}

TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_key key)
{
    struct handle* handle_struct = (struct handle*) attr;
    switch (key)
    {
    case ATTR_KEY_USE_DEVICE_MEMORY:
        *(bool*)handle_struct->attributes[key] = false;
        break;
    case ATTR_KEY_PRECISION_DIGITS:
        *(int*)handle_struct->attributes[key] = 0;
        break;
    default:
        return 15; // Invalid key
    }
    return 0;
}
