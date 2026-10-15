/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "internal.h"

#include <cstring>

TAPP_error TAPP_create_attr(TAPP_attr *attr)
{
    if (!attr) return TAPP_TBLIS_ERROR_NULL_POINTER;

    attributes* a = new attributes();
    *attr = reinterpret_cast<TAPP_attr>(a);
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_destroy_attr(TAPP_attr attr)
{
    if (attr == 0) return TAPP_TBLIS_SUCCESS;

    attributes* a = reinterpret_cast<attributes*>(attr);
    delete a;
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, size_t size, void* value)
{
    if (!attr || !value) return TAPP_TBLIS_ERROR_NULL_POINTER;

    switch (key) {
        case TAPP_THREAD_COUNT:
        {
            if (size != sizeof(int)) return TAPP_TBLIS_ERROR_INVALID_ATTR_VALUE;
            int thread_count;
            std::memcpy(&thread_count, value, sizeof(int));
            if (thread_count < 1) return TAPP_TBLIS_ERROR_INVALID_ATTR_VALUE;
        }
        break;
        case TAPP_CPU_MASK:
        {
#ifdef TAPP_TBLIS_HAS_CPU_MASK
            if (size != sizeof(cpu_set_t)) return TAPP_TBLIS_ERROR_INVALID_ATTR_VALUE;
            cpu_set_t cpu_mask;
            std::memcpy(&cpu_mask, value, sizeof(cpu_set_t));
            if (CPU_COUNT(&cpu_mask) < 1) return TAPP_TBLIS_ERROR_INVALID_ATTR_VALUE;
#else
            return TAPP_TBLIS_ERROR_CPU_MASK_UNAVAILABLE;
#endif
        }
        break;
        default:
        return TAPP_TBLIS_ERROR_UNKNOWN_ATTR_KEY;
    }
    attributes& a = *reinterpret_cast<attributes*>(attr);
    const char* bytes = static_cast<const char*>(value);
    a[key].assign(bytes, bytes + size);
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, size_t *size, void** value)
{
    if (!attr || !size || !value) return TAPP_TBLIS_ERROR_NULL_POINTER;

    attributes& a = *reinterpret_cast<attributes*>(attr);
    std::map<TAPP_key, std::vector<char>>::iterator it = a.find(key);
    if (it == a.end())
    {
        *size = 0;
        *value = nullptr;
        return TAPP_TBLIS_SUCCESS;
    }

    *size = it->second.size();
    *value = it->second.data();
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_key key)
{
    if (!attr) return TAPP_TBLIS_ERROR_NULL_POINTER;
    
    attributes& a = *reinterpret_cast<attributes*>(attr);
    a.erase(key);
    return TAPP_TBLIS_SUCCESS;
}