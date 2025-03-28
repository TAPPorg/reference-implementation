/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - November 2024
 */

#ifndef TBLIS_BIND_H
#define TBLIS_BIND_H
#ifdef __cplusplus
#include <iostream>
#include <random>
#include <tuple>
#include <string>
#include <complex>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ctf.hpp>
#pragma GCC diagnostic pop

extern "C" {
#endif
    #include "tapp.h"
#ifdef __cplusplus
    #include "tapp_ex_imp.h"
}
extern "C" {
#endif
  void ctf_bind_execute_product(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                    void* alpha, void* beta, int datatype_tapp);

  int compare_tensors_(void* A, void* B, int64_t size, int datatype_tapp);
 
  int finalizeWork();
#ifdef __cplusplus
}
#endif

#endif

