#ifndef TAPP_REF_IMPL_CUTENSOR_BINDINGS_ERROR_H_
#define TAPP_REF_IMPL_CUTENSOR_BINDINGS_ERROR_H_

#include <tapp/error.h>

#include <cutensor.h>

#include <cstring>
#include <string>

int pack_error(int current_value, int tapp_err);
int pack_error(int current_value, cutensorStatus_t e); 
int pack_error(int current_value, cudaError_t e);

#endif /* TAPP_REF_IMPL_CUTENSOR_BINDS_ERROR_H_ */