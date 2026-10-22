#include "internal.h"

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides)
{
    if (!info) return TAPP_ERR_NULL_POINTER;
    if (nmode < 0) return TAPP_ERR_NMODE_NEGATIVE;
    if (nmode > 0 && (!extents || !strides)) return TAPP_ERR_NULL_POINTER;

    for (int i = 0; i < nmode; i++)
    {
        if (extents[i] < 0) return TAPP_ERR_EXTENTS_NEGATIVE;
    }

    cutensorDataType_t cutype;
    if (!translate_datatype(type, cutype)) return TAPP_ERR_INVALID_DATATYPE;

    struct tensor_info* t = nullptr;

    try
    {
        t = new tensor_info{cutype,
                            std::vector<int64_t>(extents, extents + nmode),
                            std::vector<int64_t>(strides, strides + nmode)};
    }
    catch (const std::bad_alloc&)
    {
        return TAPP_ERR_OUT_OF_MEMORY;
    }

    *info = reinterpret_cast<TAPP_tensor_info>(t);
    return TAPP_SUCCESS;
}

TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info)
{
    if (!info) return TAPP_ERR_NULL_POINTER;
    delete reinterpret_cast<tensor_info*>(info);
    return TAPP_SUCCESS;
}

int TAPP_get_nmodes(TAPP_tensor_info info)
{
    // No error code can be returned here, so -1 signals an invalid handle
    if (!info) return -1;

    tensor_info* t = reinterpret_cast<tensor_info*>(info);
    return (int)t->extents.size();
}

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           int nmodes)
{
    if (!info) return TAPP_ERR_NULL_POINTER;
    if (nmodes < 0) return TAPP_ERR_NMODE_NEGATIVE;

    auto t = reinterpret_cast<tensor_info*>(info);

    try
    {
        t->extents.reserve(nmodes);
        t->strides.reserve(nmodes);
    }
    catch (const std::bad_alloc&)
    {
        return TAPP_ERR_OUT_OF_MEMORY;
    }
    // Existing modes are kept; new modes start with extent 0 and stride 0
    t->extents.resize(nmodes);
    t->strides.resize(nmodes);

    return TAPP_SUCCESS;
}

void TAPP_get_extents(TAPP_tensor_info info,
                      int64_t* extents)
{
    if (!info || !extents) return;

    auto t = reinterpret_cast<tensor_info*>(info);
    std::copy(t->extents.begin(), t->extents.end(), extents);
}

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            const int64_t* extents)
{
    if (!info) return TAPP_ERR_NULL_POINTER;

    auto t = reinterpret_cast<tensor_info*>(info);

    if (t->extents.size() > 0 && !extents) return TAPP_ERR_NULL_POINTER;
    for (size_t i = 0; i < t->extents.size(); i++)
    {
        if (extents[i] < 0) return TAPP_ERR_EXTENTS_NEGATIVE;
    }

    std::copy(extents, extents + t->extents.size(), t->extents.begin());
    return TAPP_SUCCESS;
}

void TAPP_get_strides(TAPP_tensor_info info,
                      int64_t* strides)
{
    if (!info || !strides) return;

    auto t = reinterpret_cast<tensor_info*>(info);
    std::copy(t->strides.begin(), t->strides.end(), strides);
}

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            const int64_t* strides)
{
    if (!info) return TAPP_ERR_NULL_POINTER;

    auto t = reinterpret_cast<tensor_info*>(info);

    if (t->strides.size() > 0 && !strides) return TAPP_ERR_NULL_POINTER;

    std::copy(strides, strides + t->strides.size(), t->strides.begin());
    return TAPP_SUCCESS;
}
