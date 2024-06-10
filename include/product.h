/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#ifndef PRODUCT_H
#define PRODUCT_H

#include <stdbool.h>
#include <complex.h>

/*
 * PRODUCT computes D <- ALPHA * A * B + BETA * C
 * IDXA: the number of dimensions of A
 * EXTA: the extent of A in each dimension
 * STRA: the stride of A in each dimension
 * A: the data of A
 * IDXB: the number of dimensions of B
 * EXTB: the extent of B in each dimension
 * STRB: the stride of B in each dimension
 * B: the data of B
 * IDXC: the number of dimensions of C
 * EXTC: the extent of C in each dimension
 * STRC: the stride of C in each dimension
 * C: the data of C
 * IDXD: the number of dimensions of D
 * EXTD: the extent of D in each dimension
 * STRD: the stride of D in each dimension
 * D: the data of D
 * ALPHA: the scalar alpha
 * BETA: the scalar beta
 * FA: whether A is transposed
 * FB: whether B is transposed
 * FC: whether C is transposed
 * EINSUM: the einsum string
 */

/*int PRODUCT(int IDXA, int* EXTA, int* STRA, float* A,
            int IDXB, int* EXTB, int* STRB, float* B,
            int IDXC, int* EXTC, int* STRC, float* C,
            int IDXD, int* EXTD, int* STRD, float* D,
            float ALPHA, float BETA, bool FA, bool FB, bool FC, char* EINSUM);*/

int PRODUCT(int IDXA, int* EXTA, int* STRA, float complex * A,
            int IDXB, int* EXTB, int* STRB, float complex* B,
            int IDXC, int* EXTC, int* STRC, float complex* C,
            int IDXD, int* EXTD, int* STRD, float complex* D,
            float complex ALPHA, float complex BETA, bool FA, bool FB, bool FC, char* EINSUM);

#endif