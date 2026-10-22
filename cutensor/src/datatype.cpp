#include "internal.h"

bool translate_datatype(TAPP_datatype in, cutensorDataType_t& out)
{
    switch (in)
    {
        case TAPP_F32:  out = CUTENSOR_R_32F;  return true;
        case TAPP_F64:  out = CUTENSOR_R_64F;  return true;
        case TAPP_C32:  out = CUTENSOR_C_32F;  return true;
        case TAPP_C64:  out = CUTENSOR_C_64F;  return true;
        // TAPP_F16 and TAPP_BF16 are not supported yet: cuTENSOR wants float scalars for them, which execute does not
        // convert to, and it refuses BF16 with the compute descriptors translate_prectype gives
        default:                               return false;
    }
}

bool translate_prectype(TAPP_prectype in, cutensorDataType_t datatype, cutensorComputeDescriptor_t& out)
{
    switch (in)
    {
        case TAPP_DEFAULT_PREC:
            switch (datatype)
            {
                case CUTENSOR_R_32F:
                case CUTENSOR_C_32F:  out = CUTENSOR_COMPUTE_DESC_32F; return true;
                case CUTENSOR_R_64F:
                case CUTENSOR_C_64F:  out = CUTENSOR_COMPUTE_DESC_64F; return true;
                default: return false; // F16 and BF16 are not supported yet, see translate_datatype
            }
        case TAPP_F32F32_ACCUM_F32: out = CUTENSOR_COMPUTE_DESC_32F; return true;
        case TAPP_F64F64_ACCUM_F64: out = CUTENSOR_COMPUTE_DESC_64F; return true;
        default: return false; // TAPP_F16F16_ACCUM_F16 waits for F16 support, see translate_datatype
    }
}

size_t sizeof_datatype(cutensorDataType_t type)
{
    switch (type)
    {
        case CUTENSOR_R_32F: return sizeof(float);
        case CUTENSOR_R_64F: return sizeof(double);
        case CUTENSOR_C_32F: return sizeof(std::complex<float>);
        case CUTENSOR_C_64F: return sizeof(std::complex<double>);
        default: return 0; // F16 and BF16 are not supported yet, see translate_datatype
    }
}