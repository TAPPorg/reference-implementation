/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "../include/tensor.h"
#include "../include/datatype.h"
#include "../include/error.h"
#include "tapp/error.h"
#include "tapp/tensor.h"

#include <algorithm>
#include <new>

TAPP_error TAPP_create_tensor_info(TAPP_tensor_info* info,
                                   TAPP_datatype type,
                                   int nmode,
                                   const int64_t* extents,
                                   const int64_t* strides) {
    if (!info) return ERROR_NULL_POINTER;
    if (nmode < 0) return ERROR_NEGATIVE_NMODE;
    if (nmode > 0 && (!extents || !strides)) return ERROR_NULL_POINTER;

    for (int i = 0; i < nmode; i++)
    {
        if (extents[i] < 0) return ERROR_NEGATIVE_EXTENT;
    }

    tblis::type_t tblis_type;
    if (!to_tblis_type(type, tblis_type)) return ERROR_UNSUPPORTED_DATATYPE;

    tensor_info* t = nullptr;

    try
    {
        t = new tensor_info{tblis_type, nmode, nullptr, nullptr};
        t->extents = new tblis::len_type[nmode];
        t->strides = new tblis::stride_type[nmode];
    }
    catch (const std::bad_alloc&)
    {
        if (t)
        {
            delete[] t->extents;
            delete t;
        }
        return ERROR_OUT_OF_MEMORY;
    }

    std::copy(extents, extents + nmode, t->extents);
    std::copy(strides, strides + nmode, t->strides);

    *info = reinterpret_cast<TAPP_tensor_info>(t);
    return SUCCESS;
}

TAPP_error TAPP_destroy_tensor_info(TAPP_tensor_info info) {
    // Destroying a null handle is a no-op, like free(NULL)
    if (!info) return SUCCESS;

    return destroy_tensor_info(reinterpret_cast<tensor_info*>(info));
}

int TAPP_get_nmodes(TAPP_tensor_info info) {
    // No error code can be returned here, so -1 signals an invalid handle
    if (!info) return -1;

    auto t = reinterpret_cast<tensor_info*>(info);
    return t->nmode;
}

TAPP_error TAPP_set_nmodes(TAPP_tensor_info info,
                           int nmodes) {
    if (!info) return ERROR_NULL_POINTER;
    if (nmodes < 0) return ERROR_NEGATIVE_NMODE;

    auto t = reinterpret_cast<tensor_info*>(info);

    // Existing modes are kept; new modes start with extent 0 and stride 0
    // and must be set by the caller with TAPP_set_extents/TAPP_set_strides
    tblis::len_type* extents = nullptr;
    tblis::stride_type* strides = nullptr;
    try
    {
        extents = new tblis::len_type[nmodes]();
        strides = new tblis::stride_type[nmodes]();
    }
    catch (const std::bad_alloc&)
    {
        delete[] extents;
        return ERROR_OUT_OF_MEMORY;
    }

    int keep = std::min(t->nmode, nmodes);
    std::copy(t->extents, t->extents + keep, extents);
    std::copy(t->strides, t->strides + keep, strides);

    delete[] t->extents;
    delete[] t->strides;
    t->extents = extents;
    t->strides = strides;
    t->nmode = nmodes;
    return SUCCESS;
}

void TAPP_get_extents(TAPP_tensor_info info,
                      int64_t* extents) {
    if (!info || !extents) return;

    auto t = reinterpret_cast<tensor_info*>(info);
    std::copy(t->extents, t->extents + t->nmode, extents);
}

TAPP_error TAPP_set_extents(TAPP_tensor_info info,
                            const int64_t* extents) {
    if (!info) return ERROR_NULL_POINTER;

    auto t = reinterpret_cast<tensor_info*>(info);

    if (t->nmode > 0 && !extents) return ERROR_NULL_POINTER;
    for (int i = 0; i < t->nmode; i++)
    {
        if (extents[i] < 0) return ERROR_NEGATIVE_EXTENT;
    }

    std::copy(extents, extents + t->nmode, t->extents);
    return SUCCESS;
}

void TAPP_get_strides(TAPP_tensor_info info,
                      int64_t* strides) {
    if (!info || !strides) return;

    auto t = reinterpret_cast<tensor_info*>(info);
    std::copy(t->strides, t->strides + t->nmode, strides);
}

TAPP_error TAPP_set_strides(TAPP_tensor_info info,
                            const int64_t* strides) {
    if (!info) return ERROR_NULL_POINTER;

    auto t = reinterpret_cast<tensor_info*>(info);

    if (t->nmode > 0 && !strides) return ERROR_NULL_POINTER;

    std::copy(strides, strides + t->nmode, t->strides);
    return SUCCESS;
}

tensor_info* clone_tensor_info(tensor_info* info) {
    if (!info) return 0;

    tensor_info* t = nullptr;
    try
    {
        t = new tensor_info{info->type, info->nmode, nullptr, nullptr, info->conj};
        t->extents = new tblis::len_type[info->nmode];
        t->strides = new tblis::stride_type[info->nmode];
    }
    catch (const std::bad_alloc&)
    {
        if (t) delete[] t->extents;
        delete t;
        return 0;
    }

    std::copy(info->extents, info->extents + info->nmode, t->extents);
    std::copy(info->strides, info->strides + info->nmode, t->strides);
    return t;
}

TAPP_error destroy_tensor_info(tensor_info* info) {
    delete[] info->extents;
    delete[] info->strides;
    delete info;

    return SUCCESS;
}

tblis::tblis_tensor* create_tblis_tensor(const tensor_info* info) {
    if (!info) return nullptr;

    tblis::tblis_tensor* t = nullptr;
    try
    {
        t = new tblis::tblis_tensor;
        t->len = new tblis::len_type[info->nmode];
        t->stride = new tblis::stride_type[info->nmode];
    }
    catch (const std::bad_alloc&)
    {
        if (t) delete[] t->len;
        delete t;
        return nullptr;
    }

    t->type = info->type;
    t->conj = info->conj;
    // reset, not operator=: assignment converts into the existing (double) type and keeps it
    t->scalar.reset(tblis::tblis_scalar(1.0, info->type));
    t->ndim = info->nmode;
    std::copy(info->extents, info->extents + info->nmode, t->len);
    std::copy(info->strides, info->strides + info->nmode, t->stride);
    return t;
}

void free_tblis_tensor(tblis::tblis_tensor* t) {
    if (!t) return;
    delete[] t->len;
    delete[] t->stride;
    delete t;
}
