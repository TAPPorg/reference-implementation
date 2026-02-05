#include "cutensor_bind.h"

cutensorDataType_t translate_datatype(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
        return CUTENSOR_R_32F;
        break;
    case TAPP_F64:
        return CUTENSOR_R_64F;
        break;
    case TAPP_C32:
        return CUTENSOR_C_32F;
        break;
    case TAPP_C64:
        return CUTENSOR_C_64F;
        break;
    case TAPP_F16:
        return CUTENSOR_R_16F;
        break;
    case TAPP_BF16:
        return CUTENSOR_R_16BF;
        break;
    default: // TODO: Default should probably be an error
        return CUTENSOR_R_32F;
        break;
    }
}

cutensorComputeDescriptor_t translate_prectype(TAPP_prectype prec, TAPP_datatype datatype)
{
    switch (prec)
    {
        case TAPP_DEFAULT_PREC:
            switch (datatype)
            {
            case TAPP_F32:
            case TAPP_C32:
                return CUTENSOR_COMPUTE_DESC_32F;
                break;
            case TAPP_F64:
            case TAPP_C64:
                return CUTENSOR_COMPUTE_DESC_64F;
                break;
            default: // TODO: Default should probably be an error
                return CUTENSOR_COMPUTE_DESC_32F;
                break;
            }
            break;
        case TAPP_F32F32_ACCUM_F32:
            return CUTENSOR_COMPUTE_DESC_32F;
            break;
        case TAPP_F64F64_ACCUM_F64:
            return CUTENSOR_COMPUTE_DESC_64F;
            break;
        case TAPP_F16F16_ACCUM_F16:
            return CUTENSOR_COMPUTE_DESC_16F;
            break;
        default: // TODO: Default should probably be an error
            return CUTENSOR_COMPUTE_DESC_32F;
            break;
    }
}

size_t sizeof_datatype(TAPP_datatype type)
{
    switch (type)
    {
    case TAPP_F32:
        return sizeof(float);
        break;
    case TAPP_F64:
        return sizeof(double);
        break;
    case TAPP_C32: 
        return sizeof(std::complex<float>);
        break;
    case TAPP_C64:
        return sizeof(std::complex<double>);
        break;
    /*case TAPP_F16: // Fix these datatypes
        //return _Float16;
        break;
    case TAPP_BF16:
        //return __bf16;
        break;*/
    default: // TODO: Default should probably be an error
        return sizeof(float);
        break;
    }
}