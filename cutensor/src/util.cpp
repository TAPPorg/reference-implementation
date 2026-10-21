#include "internal.h"

int64_t compute_index(const int64_t* coordinates, int nmode, const int64_t* strides)
{
    int64_t index = 0;
    for (int i = 0; i < nmode; i++)
    {
        index += coordinates[i] * strides[i];
    }
    return index;

}

void increment_coordinates(int64_t* coordinates, int nmode, const int64_t* extents)
{
    if (nmode <= 0)
    {
        return;
    }

    int k = 0;
    do
    {
        coordinates[k] = (coordinates[k] + 1) % extents[k];
        k++;
    } while (coordinates[k - 1] == 0 && k < nmode);
}

cutensorOperator_t translate_operator(TAPP_element_op op)
{
    switch (op)
    {
    case TAPP_IDENTITY:
        return CUTENSOR_OP_IDENTITY;
        break;
    case TAPP_CONJUGATE:
        return CUTENSOR_OP_CONJ;
        break;
    default: // TODO: Default should probably be an error
        return CUTENSOR_OP_IDENTITY;
        break;
    }
}

bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B)
{
    if (A->nmode != B->nmode) return false;
    for (int i = 0; i < A->nmode; i++)
    {
        if (A->extents[i] != B->extents[i] || A->strides[i] != B->strides[i] || idx_A[i] != idx_B[i]) return false;
    }
    return true;
}

bool is_zero_scalar(TAPP_datatype type, const void* value)
{
    switch (type)
    {
    case TAPP_F32: return *(const float*)value == 0.0f;
    case TAPP_F64: return *(const double*)value == 0.0;
    case TAPP_C32: return *(const std::complex<float>*)value == std::complex<float>(0, 0);
    case TAPP_C64: return *(const std::complex<double>*)value == std::complex<double>(0, 0);
    default:       return false;
    }
}

void load_scalar(TAPP_datatype type, const void* value, bool conj, void* out)
{
    switch (type)
    {
    case TAPP_C32:
    {
        std::complex<float> v = *(const std::complex<float>*)value;
        if (conj) v = std::conj(v);
        memcpy(out, &v, sizeof(v));
        break;
    }
    case TAPP_C64:
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
