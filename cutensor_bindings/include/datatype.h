#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_DATATYPE_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_DATATYPE_H_

#include <tapp/datatype.h>

#include <cutensor.h>

#include <complex>

cutensorDataType_t translate_datatype(TAPP_datatype type);

cutensorComputeDescriptor_t translate_prectype(TAPP_prectype prec, TAPP_datatype datatype);

size_t sizeof_datatype(TAPP_datatype type);

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDINGS_DATATYPE_H_ */