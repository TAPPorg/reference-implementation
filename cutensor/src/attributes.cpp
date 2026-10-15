#include "../include/attributes.h"
#include "../include/executor.h"

TAPP_EXPORT TAPP_error TAPP_create_attr(TAPP_attr *attr){
    Attributes* attr_ptr = new Attributes();
    *attr = reinterpret_cast<TAPP_attr>(attr_ptr);

    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_attr(TAPP_attr attr)
{
    if (attr == 0) return TAPP_SUCCESS;

    Attributes* a = reinterpret_cast<Attributes*>(attr);
    delete a;
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_set(TAPP_attr attr, TAPP_key key, size_t size, void* value)
{
    if (!attr || !value) return TAPP_ERR_NULL_POINTER;

    switch (key) {
        case TAPP_CUTENSOR_DEVICE:
        {
            if (size != sizeof(int)) return TAPP_ERR_ATTR_INVALID_VALUE;
            int device_id;
            std::memcpy(&device_id, value, sizeof(int));
            if (device_id < 0) return TAPP_ERR_ATTR_INVALID_VALUE;
        }
        break;
        case TAPP_CUTENSOR_STREAM:
        {
            if (size != sizeof(cudaStream_t)) return TAPP_ERR_ATTR_INVALID_VALUE;
            cudaStream_t stream = *static_cast<cudaStream_t*>(value);
            if((*((cudaStream_t *) value))==nullptr) break;
            unsigned int flags = 0;
            cudaError_t status = cudaStreamGetFlags(stream, &flags);
            if(status!=cudaSuccess) return TAPP_ERR_ATTR_INVALID_VALUE;
        }
        break;
        default:
        return TAPP_ERR_ATTR_UNKNOWN_KEY;
    }
    Attributes& a = *reinterpret_cast<Attributes*>(attr);
    const char* bytes = static_cast<const char*>(value);

    a[key].assign(bytes, bytes + size);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_attr_get(TAPP_attr attr, TAPP_key key, size_t *size, void** value)
{
    if (!attr || !size || !value) return TAPP_ERR_NULL_POINTER;

    Attributes& a = *reinterpret_cast<Attributes*>(attr);
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

    Attributes& a = *reinterpret_cast<Attributes*>(attr);
    if(key==TAPP_CUTENSOR_STREAM){
        cudaStream_t stream;
        size_t size;
        TAPP_attr_get(attr, TAPP_CUTENSOR_STREAM, &size, (void**) &stream);

        if(stream!=nullptr){
            cudaError_t status = cudaStreamDestroy(stream);
            if(status!=cudaSuccess) return TAPP_ERR_ATTR_INVALID_STREAM;
        }
    }

    a.erase(key);
    return TAPP_SUCCESS;
}
