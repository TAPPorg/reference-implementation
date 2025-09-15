#include "tapp_ex_imp.h"
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

float* tucker_to_tensor_contraction(int nmode_A, int64_t* extents_A, int64_t* strides_A, float* A,
                                    int nmode_B, int64_t* extents_B, int64_t* strides_B, float* B,
                                    int nmode_D, int64_t* extents_D, int64_t* strides_D, float* D,
                                    int64_t* idx_A, int64_t* idx_B, int64_t* idx_D);