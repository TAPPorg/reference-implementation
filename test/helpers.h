/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */

#ifdef __cplusplus
#include <complex>
#include <cstdint>
extern "C" {
#else
#include <complex.h>
#include <stdint.h>
#endif

void print_tensor_s(int nmode, const int64_t *extents, const int64_t *strides, const float *data);

#ifdef __cplusplus
// std::complex<float> is layout-compatible with C's float complex
void print_tensor_c(int nmode, const int64_t *extents, const int64_t *strides, const std::complex<float> *data);
#else
void print_tensor_c(int nmode, const int64_t *extents, const int64_t *strides, const float complex *data);
#endif

#ifdef __cplusplus
}
#endif
