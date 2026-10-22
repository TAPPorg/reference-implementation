#include "internal.h"

bool translate_operator(TAPP_element_op in, cutensorOperator_t& out)
{
    switch (in)
    {
        case TAPP_IDENTITY:  out = CUTENSOR_OP_IDENTITY; return true;
        case TAPP_CONJUGATE: out = CUTENSOR_OP_CONJ;     return true;
        default: return false;
    }
}

bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B)
{
    if (A->extents.size() != B->extents.size()) return false;
    for (size_t i = 0; i < A->extents.size(); i++)
    {
        if (A->extents[i] != B->extents[i] || A->strides[i] != B->strides[i] || idx_A[i] != idx_B[i]) return false;
    }
    return true;
}

bool is_zero_scalar(cutensorDataType_t type, const void* value)
{
    switch (type)
    {
    case CUTENSOR_R_32F: return *(const float*)value == 0.0f;
    case CUTENSOR_R_64F: return *(const double*)value == 0.0;
    case CUTENSOR_C_32F: return *(const std::complex<float>*)value == std::complex<float>(0, 0);
    case CUTENSOR_C_64F: return *(const std::complex<double>*)value == std::complex<double>(0, 0);
    default:       return false;
    }
}

void load_scalar(cutensorDataType_t type, const void* value, bool conj, void* out)
{
    switch (type)
    {
    case CUTENSOR_C_32F:
    {
        std::complex<float> v = *(const std::complex<float>*)value;
        if (conj) v = std::conj(v);
        memcpy(out, &v, sizeof(v));
        break;
    }
    case CUTENSOR_C_64F:
    {
        std::complex<double> v = *(const std::complex<double>*)value;
        if (conj) v = std::conj(v);
        memcpy(out, &v, sizeof(v));
        break;
    }
    default: // Real types: conjugation does nothing
        memcpy(out, value, sizeof_datatype(type));
        break;
    }
}

bool is_aligned(const void* p)
{
    return p == nullptr || uintptr_t(p) % kAlignment == 0;
}

bool is_valid_op(TAPP_element_op op)
{
    return op == TAPP_IDENTITY || op == TAPP_CONJUGATE;
}
