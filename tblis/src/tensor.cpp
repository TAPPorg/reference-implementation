/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "internal.h"
#include "tblis/frame/base/basic_types.h"

#include <algorithm>
#include <cstddef>
#include <new>

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides) {
    if (!info) return TAPP_TBLIS_ERROR_NULL_POINTER;
    if (nmode < 0) return TAPP_TBLIS_ERROR_NEGATIVE_NMODE;
    if (nmode > 0 && (!extents || !strides)) return TAPP_TBLIS_ERROR_NULL_POINTER;

    for (int i = 0; i < nmode; i++)
    {
        if (extents[i] < 0) return TAPP_TBLIS_ERROR_NEGATIVE_EXTENT;
    }

    tblis::type_t tblis_type;
    if (!to_tblis_type(type, tblis_type)) return TAPP_TBLIS_ERROR_UNSUPPORTED_DATATYPE;

    tensor_info* t = nullptr;

    try
    {
        t = new tensor_info{tblis_type,
                            std::vector<tblis::len_type>(extents, extents + nmode),
                            std::vector<tblis::stride_type>(strides, strides + nmode),
                            0};
    }
    catch (const std::bad_alloc&)
    {
        return TAPP_TBLIS_ERROR_OUT_OF_MEMORY;
    }

    *info = reinterpret_cast<TAPP_tensor_info>(t);
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info) {
    // Destroying a null handle is a no-op, like free(NULL)
    if (!info) return TAPP_TBLIS_SUCCESS;
    delete reinterpret_cast<tensor_info*>(info);
    return TAPP_TBLIS_SUCCESS;
}

int TAPP_get_nmodes(TAPP_tensor_info info) {
    // No error code can be returned here, so -1 signals an invalid handle
    if (!info) return -1;

    tensor_info* t = reinterpret_cast<tensor_info*>(info);
    return (int)t->extents.size();
}

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           int nmodes) {
    if (!info) return TAPP_TBLIS_ERROR_NULL_POINTER;
    if (nmodes < 0) return TAPP_TBLIS_ERROR_NEGATIVE_NMODE;

    auto t = reinterpret_cast<tensor_info*>(info);

    try
    {
        t->extents.reserve(nmodes);
        t->strides.reserve(nmodes);
    }
    catch (const std::bad_alloc&)
    {
        return TAPP_TBLIS_ERROR_OUT_OF_MEMORY;
    }
    // Existing modes are kept; new modes start with extent 0 and stride 0
    t->extents.resize(nmodes);
    t->strides.resize(nmodes);

    return TAPP_TBLIS_SUCCESS;
}

void TAPP_get_extents(TAPP_tensor_info info,
                      int64_t* extents) {
    if (!info || !extents) return;

    auto t = reinterpret_cast<tensor_info*>(info);
    std::copy(t->extents.begin(), t->extents.end(), extents);
}

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            const int64_t* extents) {
    if (!info) return TAPP_TBLIS_ERROR_NULL_POINTER;

    auto t = reinterpret_cast<tensor_info*>(info);

    if (t->extents.size() > 0 && !extents) return TAPP_TBLIS_ERROR_NULL_POINTER;
    for (size_t i = 0; i < t->extents.size(); i++)
    {
        if (extents[i] < 0) return TAPP_TBLIS_ERROR_NEGATIVE_EXTENT;
    }

    std::copy(extents, extents + t->extents.size(), t->extents.begin());
    return TAPP_TBLIS_SUCCESS;
}

void TAPP_get_strides(TAPP_tensor_info info,
                      int64_t* strides) {
    if (!info || !strides) return;

    auto t = reinterpret_cast<tensor_info*>(info);
    std::copy(t->strides.begin(), t->strides.end(), strides);
}

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            const int64_t* strides) {
    if (!info) return TAPP_TBLIS_ERROR_NULL_POINTER;

    auto t = reinterpret_cast<tensor_info*>(info);

    if (t->strides.size() > 0 && !strides) return TAPP_TBLIS_ERROR_NULL_POINTER;

    std::copy(strides, strides + t->strides.size(), t->strides.begin());
    return TAPP_TBLIS_SUCCESS;
}


tblis::tblis_tensor create_tblis_tensor(const tensor_info& info,
                                        const void* data,
                                        const tblis::tblis_scalar* scalar,
                                        const bool* conj) {

    tblis::tblis_tensor t;

    t.ndim = (int)info.extents.size();
    t.len = const_cast<tblis::len_type*>(info.extents.data());
    t.stride = const_cast<tblis::stride_type*>(info.strides.data());
    t.data = const_cast<void*>(data);
    t.type = info.type;
    t.conj = (conj == nullptr ? info.conj : *conj);
    // reset, not operator=: assignment converts into the existing (double) type and keeps it
    t.scalar.reset(scalar == nullptr ? tblis::tblis_scalar(1.0, info.type) : *scalar);
    return t;
}
