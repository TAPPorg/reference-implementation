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
  int distributed_broadcast_tensor_body_data(TAPP_tensor_info info, void* data);
  int distributed_gather_tensor_body_data(TAPP_tensor_info info, void* data);

  int ctf_bind_execute_product(TAPP_tensor_info info_A, void* A, int op_A, int64_t* idx_A,
                    TAPP_tensor_info info_B, void* B, int op_B, int64_t* idx_B,
                    TAPP_tensor_info info_C, void* C, int op_C, int64_t* idx_C,
                    TAPP_tensor_info info_D, void* D, int op_D, int64_t* idx_D,
                    void* alpha, void* beta);

  int compare_tensors_(void* A, void* B, int64_t size, int datatype_tapp);
  int distributed_get_uuid_len();
  int distributed_get_uuid(char * uuid, const int uuid_len);
  int distributed_destruct_tensor(TAPP_tensor_info info);

  int distributed_tensor_set_name(TAPP_tensor_info info, const int64_t* name, const int name_len);
  int distributed_initialize_tensor(TAPP_tensor_info info, void* init_val);
  int distributed_copy_tensor(TAPP_tensor_info dest, TAPP_tensor_info src);
  int distributed_scale_with_denominators(TAPP_tensor_info info, 
		                  const int n_occ, const int n_vir, void* eps_occ, void* eps_vir, void* eps_ijk);

  int finalizeWork();
#ifdef __cplusplus
}
#endif

#endif

