#ifndef TAPP_TBLIS_DATATYPE_H_
#define TAPP_TBLIS_DATATYPE_H_

#pragma GCC diagnostic push
#include "tblis.h"
#pragma GCC diagnostic pop

#include <tapp.h>

/*
 * Convert a TAPP storage datatype to the corresponding TBLIS type.
 * Returns false if the datatype has no TBLIS equivalent (e.g. TAPP_F16,
 * TAPP_BF16), in which case out is left unchanged.
 *
 * Note: the numeric values differ between the two libraries
 * (TAPP_F64 = 1 but TYPE_DOUBLE = 2, TAPP_C32 = 2 but TYPE_SCOMPLEX = 1),
 * so the values must never be cast directly.
 */
inline bool to_tblis_type(TAPP_datatype in, tblis::type_t& out)
{
    switch (in)
    {
        case TAPP_F32: out = tblis::TYPE_FLOAT;    return true;
        case TAPP_F64: out = tblis::TYPE_DOUBLE;   return true;
        case TAPP_C32: out = tblis::TYPE_SCOMPLEX; return true;
        case TAPP_C64: out = tblis::TYPE_DCOMPLEX; return true;
        default:                                   return false;
    }
}

/*
 * Size in bytes of one element of the given TBLIS type, 0 if the type is unknown.
 */
inline size_t tblis_type_size(tblis::type_t type)
{
    switch (type)
    {
        case tblis::TYPE_FLOAT:    return sizeof(float);
        case tblis::TYPE_DOUBLE:   return sizeof(double);
        case tblis::TYPE_SCOMPLEX: return sizeof(tblis::scomplex);
        case tblis::TYPE_DCOMPLEX: return sizeof(tblis::dcomplex);
        default:                   return 0;
    }
}

/*
 * Read a scalar of the given TBLIS type from untyped memory, as passed for
 * alpha and beta through the TAPP API. The type must be one returned by
 * to_tblis_type.
 */
inline tblis::tblis_scalar to_tblis_scalar(tblis::type_t type, const void* value)
{
    switch (type)
    {
        case tblis::TYPE_FLOAT:    return tblis::tblis_scalar(*static_cast<const float*>(value));
        case tblis::TYPE_DOUBLE:   return tblis::tblis_scalar(*static_cast<const double*>(value));
        case tblis::TYPE_SCOMPLEX: return tblis::tblis_scalar(*static_cast<const tblis::scomplex*>(value));
        case tblis::TYPE_DCOMPLEX: return tblis::tblis_scalar(*static_cast<const tblis::dcomplex*>(value));
        default:                   return tblis::tblis_scalar(0.0);  // unreachable for valid types, tblis_scalar has no default constructor
    }
}

#endif  /* TAPP_TBLIS_DATATYPE_H_ */
