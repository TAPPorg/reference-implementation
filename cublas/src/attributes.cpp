#include "internal.h"
#include "tapp_cublas.h"

TAPP_EXPORT TAPP_error TAPP_create_attr(TAPP_attr *attr){
    if (!attr) return TAPP_ERR_NULL_POINTER;

    attributes* attr_ptr = new attributes();
    *attr = reinterpret_cast<TAPP_attr>(attr_ptr);

    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_attr(TAPP_attr attr)
{
    if (attr == 0) return TAPP_SUCCESS;

    attributes* a = reinterpret_cast<attributes*>(attr);
    delete a;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, size_t size, void* value)
{
    if (!attr || !value) return TAPP_ERR_NULL_POINTER;

    switch (key) {
        case TAPP_CUBLAS_EMULATION_STRATEGY:
        {
            if (size != sizeof(TAPP_emulation_strategy)) return TAPP_ERR_ATTR_INVALID_VALUE;
            TAPP_emulation_strategy strategy;
            std::memcpy(&strategy, value, sizeof(TAPP_emulation_strategy));
            if (strategy < 0) return TAPP_ERR_ATTR_INVALID_VALUE;
        }
        break;
        case TAPP_CUBLAS_EMULATION_MANTISSA_CONTROL:
        {
            if (size != sizeof(TAPP_emulation_mantissa_control)) return TAPP_ERR_ATTR_INVALID_VALUE;
            TAPP_emulation_mantissa_control mantissa_control;
            std::memcpy(&mantissa_control, value, sizeof(TAPP_emulation_mantissa_control));
            if (mantissa_control < 0) return TAPP_ERR_ATTR_INVALID_VALUE;
        }
        break;
        case TAPP_CUBLAS_EMULATION_NUM_GEMM:
        {
            if (size != sizeof(TAPP_num_gemm)) return TAPP_ERR_ATTR_INVALID_VALUE;
            TAPP_num_gemm num_gemm;
            std::memcpy(&num_gemm, value, sizeof(TAPP_num_gemm));
            if (num_gemm < TAPP_2_GEMM || num_gemm > TAPP_20_GEMM) return TAPP_ERR_ATTR_INVALID_VALUE;
        }
        break;
        default:
        return TAPP_ERR_ATTR_UNKNOWN_KEY;
    }
    attributes& a = *reinterpret_cast<attributes*>(attr);
    const char* bytes = static_cast<const char*>(value);

    a[key].assign(bytes, bytes + size);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, size_t *size, void** value)
{
    if (!attr || !size || !value) return TAPP_ERR_NULL_POINTER;

    attributes& a = *reinterpret_cast<attributes*>(attr);
    std::map<TAPP_key, std::vector<char>>::iterator it = a.find(key);
    if (it == a.end())
    {
        *size = 0;
        *value = nullptr;
        return TAPP_SUCCESS;
    }

    *size = it->second.size();
    *value = it->second.data();
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_clear(TAPP_attr attr, TAPP_key key)
{
    if (!attr) return TAPP_ERR_NULL_POINTER;

    attributes& a = *reinterpret_cast<attributes*>(attr);
    a.erase(key);
    return TAPP_SUCCESS;
}
