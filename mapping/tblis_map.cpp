/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - July 2024
 */

#include <iostream>
#include <random>
#include <tuple>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tblis.h"
#pragma GCC diagnostic pop
extern "C" {
    #include "product.h"
}

void tensor_mult(const tblis_comm* comm, const tblis_config* cfg,
                 const tblis_tensor* A, const label_type* idx_A,
                 const tblis_tensor* B, const label_type* idx_B,
                 tblis_tensor* C, const label_type* idx_C) {
    int IDXA = A->ndim;
    int* EXTA = A->len;
    int* STRA = A->stride;
    float* data_A = static_cast<float*>(A->data);

    int IDXB = B->ndim;
    int* EXTB = B->len;
    int* STRB = B->stride;
    float* data_B = static_cast<float*>(B->data);

    int IDXD = C->ndim;
    int* EXTD = C->len;
    int* STRD = C->stride;
    float* data_D = static_cast<float*>(C->data);

    int IDXC = IDXD;
    int* EXTC = EXTD;
    int* STRC = STRD;
    float* data_C = data_D;

    char* EINSUM = (char*)malloc((IDXA + IDXB + IDXD + 1 + 2 + 1) * sizeof(char));

    for (int i = 0; i < IDXA; i++) {
        EINSUM[i] = idx_A[i];
    }
    EINSUM[IDXA] = ',';
    for (int i = 0; i < IDXB; i++) {
        EINSUM[IDXA + 1 + i] = idx_B[i];
    }
    EINSUM[IDXA + 1 + IDXB] = '-';
    EINSUM[IDXA + 1 + IDXB + 1] = '>';
    for (int i = 0; i < IDXD; i++) {
        EINSUM[IDXA + 1 + IDXB + 2 + i] = idx_C[i];
    }
    EINSUM[IDXA + 1 + IDXB + 2 + IDXD] = '\0';

    float ALPHA = 1.0;

    float BETA = 0.0;

    bool FA = false;

    bool FB = false;

    bool FC = false;

    PRODUCT(IDXA, EXTA, STRA, data_A,
            IDXB, EXTB, STRB, data_B,
            IDXC, EXTC, STRC, data_C,
            IDXD, EXTD, STRD, data_D,
            ALPHA, BETA, FA, FB, FC, EINSUM);
}