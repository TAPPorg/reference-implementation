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
   
  int distributed_create_tensor(TAPP_tensor_info info, void* init_val);

  int ctf_bind_execute_product(TAPP_tensor_info info_A, void* A, int op_A, int64_t* idx_A,
                    TAPP_tensor_info info_B, void* B, int op_B, int64_t* idx_B,
                    TAPP_tensor_info info_C, void* C, int op_C, int64_t* idx_C,
                    TAPP_tensor_info info_D, void* D, int op_D, int64_t* idx_D,
                    void* alpha, void* beta);

  int compare_tensors_(void* A, void* B, int64_t size, int datatype_tapp);
  int distributed_get_uuid_len();
  int distributed_get_uuid(char * uuid, const int uuid_len);
  int distributed_destruct_tensor(TAPP_tensor_info info);
  int finalizeWork();
#ifdef __cplusplus
}
#endif

#endif

