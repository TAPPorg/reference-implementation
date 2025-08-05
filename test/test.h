/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - November 2024
 */
#include <iostream>
#include <random>
#include <tuple>
#include <string>
#include <complex>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tblis.h"
#pragma GCC diagnostic pop
extern "C" {
    #include "hi_tapp.h"
    #include "hi_tapp/tapp_ex_imp.h"
}

void run_tblis_mult_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, float* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, float* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, float* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, float* D, int op_D, int64_t* idx_D,
                    float alpha, float beta);
bool compare_tensors_s(float* A, float* B, int size);
std::tuple<int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           float, float,
           float*, float*, float*, float*,
           int64_t, int64_t, int64_t, int64_t> generate_contraction_s(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, int min_extent,
                                                                       bool equal_extents, bool lower_extents,
                                                                       bool lower_idx, bool negative_str,
                                                                       bool unique_idx, bool repeated_idx,
                                                                       bool mixed_str);
float rand_s(float min, float max);
float rand_s();
void print_tensor_s(int nmode, int64_t* extents, int64_t* strides, float* data);
std::tuple<float*, float*> copy_tensor_data_s(int64_t size, float* data, float* pointer);
float* copy_tensor_data_s(int size, float* data);
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, float*> contract_unique_idx_s(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2);
float* create_tensor_data_s(int64_t size);

void run_tblis_mult_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, double* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, double* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, double* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, double* D, int op_D, int64_t* idx_D,
                    double alpha, double beta);
bool compare_tensors_d(double* A, double* B, int size);
std::tuple<int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           double, double,
           double*, double*, double*, double*,
           int64_t, int64_t, int64_t, int64_t> generate_contraction_d(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, int min_extent,
                                                                       bool equal_extents, bool lower_extents,
                                                                       bool lower_idx, bool negative_str,
                                                                       bool unique_idx, bool repeated_idx,
                                                                       bool mixed_str);
double rand_d(double min, double max);
double rand_d();
void print_tensor_d(int nmode, int64_t* extents, int64_t* strides, double* data);
float* copy_tensor_data_d(int size, float* data);
std::tuple<double*, double*> copy_tensor_data_d(int64_t size, double* data, double* pointer);
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, double*> contract_unique_idx_d(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2);
double* create_tensor_data_d(int64_t size);

void run_tblis_mult_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<float>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<float>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<float>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<float>* D, int op_D, int64_t* idx_D,
                    std::complex<float> alpha, std::complex<float> beta);
bool compare_tensors_c(std::complex<float>* A, std::complex<float>* B, int size);
std::tuple<int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           std::complex<float>, std::complex<float>,
           std::complex<float>*, std::complex<float>*, std::complex<float>*, std::complex<float>*,
           int64_t, int64_t, int64_t, int64_t> generate_contraction_c(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, int min_extent,
                                                                       bool equal_extents, bool lower_extents,
                                                                       bool lower_idx, bool negative_str,
                                                                       bool unique_idx, bool repeated_idx,
                                                                       bool mixed_str);
std::complex<float> rand_c(std::complex<float> min, std::complex<float> max);
std::complex<float> rand_c();
void print_tensor_c(int nmode, int64_t* extents, int64_t* strides, std::complex<float>* data);
float* copy_tensor_data_c(int size, float* data);
std::tuple<std::complex<float>*, std::complex<float>*> copy_tensor_data_c(int64_t size, std::complex<float>* data, std::complex<float>* pointer);
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, std::complex<float>*> contract_unique_idx_c(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2);
std::complex<float>* create_tensor_data_c(int64_t size);

void run_tblis_mult_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<double>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<double>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<double>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<double>* D, int op_D, int64_t* idx_D,
                    std::complex<double> alpha, std::complex<double> beta);
bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size);
std::tuple<int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           std::complex<double>, std::complex<double>,
           std::complex<double>*, std::complex<double>*, std::complex<double>*, std::complex<double>*,
           int64_t, int64_t, int64_t, int64_t> generate_contraction_z(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, int min_extent,
                                                                       bool equal_extents, bool lower_extents,
                                                                       bool lower_idx, bool negative_str,
                                                                       bool unique_idx, bool repeated_idx,
                                                                       bool mixed_str);
std::complex<double> rand_z(std::complex<double> min, std::complex<double> max);
std::complex<double> rand_z();
void print_tensor_z(int nmode, int64_t* extents, int64_t* strides, std::complex<double>* data);
float* copy_tensor_data_z(int size, float* data);
std::tuple<std::complex<double>*, std::complex<double>*> copy_tensor_data_z(int64_t size, std::complex<double>* data, std::complex<double>* pointer);
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, std::complex<double>*> contract_unique_idx_z(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2);
std::complex<double>* create_tensor_data_z(int64_t size);



std::string str(bool b);
int randi(int min, int max);
char* swap_indices(char* indices, int nmode_A, int nmode_B, int nmode_D);
void add_incorrect_idx(int64_t max_idx, int* nmode, int64_t** idx, int64_t** extents, int64_t** strides);
tblis::len_type* translate_extents_to_tblis(int nmode, int64_t* extents);
tblis::stride_type* translate_strides_to_tblis(int nmode, int64_t* strides);
tblis::label_type* translate_idx_to_tblis(int nmode, int64_t* idx);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
int* choose_stride_signs(int nmode, bool negative_str, bool mixed_str);
bool* choose_subtensor_dims(int nmode, int outer_nmode);
int64_t* calculate_outer_extents(int outer_nmode, int64_t* extents, bool* subtensor_dims, bool lower_extents);
int64_t* calculate_offsets(int nmode, int outer_nmode, int64_t* extents, int64_t* outer_extents, bool* subtensor_dims, bool lower_extents);
int64_t* calculate_strides(int nmode, int outer_nmode, int64_t* outer_extents, int* stride_signs, bool* subtensor_dims);
int calculate_size(int nmode, int64_t* extents);
void* calculate_tensor_pointer(void* pointer, int nmode, int64_t* extents, int64_t* offsets, int64_t* strides, unsigned long data_size);

// Tests
bool test_hadamard_product();
bool test_contraction();
bool test_commutativity();
bool test_permutations();
bool test_equal_extents();
bool test_outer_product();
bool test_full_contraction();
bool test_zero_dim_tensor_contraction();
bool test_one_dim_tensor_contraction();
bool test_subtensor_same_idx();
bool test_subtensor_lower_idx();
bool test_negative_strides();
bool test_negative_strides_subtensor_same_idx();
bool test_negative_strides_subtensor_lower_idx();
bool test_mixed_strides();
bool test_mixed_strides_subtensor_same_idx();
bool test_mixed_strides_subtensor_lower_idx();
bool test_contraction_double_precision();
bool test_contraction_complex();
bool test_contraction_complex_double_precision();
bool test_zero_stride();
bool test_unique_idx();
bool test_repeated_idx();
bool test_hadamard_and_free();
bool test_hadamard_and_contraction();
bool test_error_non_matching_ext();
bool test_error_C_other_structure();
bool test_error_aliasing_within_D();
