/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - November 2024
 */

#ifndef TAPP_REF_IMPL_TBLIS_BIND_TBLIS_BIND_H_
#define TAPP_REF_IMPL_TBLIS_BIND_TBLIS_BIND_H_

#ifdef __cplusplus
    #pragma GCC diagnostic push
    //#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #include "tblis.h"
    #pragma GCC diagnostic pop
#endif

#include <tapp.h>

TAPP_EXPORT void
bind_tblis_execute_product(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                    void* alpha, void* beta, int datatype_tapp);

TAPP_EXPORT int compare_tensors_(void* A, void* B, int64_t size, int datatype_tapp);

#endif /* TAPP_REF_IMPL_TBLIS_BIND_TBLIS_BIND_H_ */

