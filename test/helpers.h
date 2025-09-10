/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */

#include <complex.h>
#include <stdint.h>

void print_tensor_s(int nmode, int64_t *extents, int64_t *strides, float *data);
void print_tensor_c(int nmode, int64_t *extents, int64_t *strides, float complex *data);