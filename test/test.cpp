/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
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
    #include "tapp.h"
    #include "tapp_ex_imp.h"
}

void run_tblis_mult_s(int nmode_A, int* extents_A, int* strides_A, float* A, int op_A, char* idx_A,
                    int nmode_B, int* extents_B, int* strides_B, float* B, int op_B, char* idx_B,
                    int nmode_C, int* extents_C, int* strides_C, float* C, int op_C, char* idx_C,
                    int nmode_D, int* extents_D, int* strides_D, float* D, int op_D, char* idx_D,
                    float alpha, float beta);

bool compare_tensors_s(float* A, float* B, int size);
bool compare_tensors_d(double* A, double* B, int size);
bool compare_tensors_c(std::complex<float>* A, std::complex<float>* B, int size);
bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size);

void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
void conjugate_c(int nmode, int64_t* extents, int64_t* strides, std::complex<float>* T);
void conjugate_z(int nmode, int64_t* extents, int64_t* strides, std::complex<double>* T);

std::tuple<int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           float, float,
           float*, float*, float*, float*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction_s(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, bool equal_extents,
                                                                       bool lower_extents, bool lower_idx,
                                                                       bool negative_str);

std::tuple<int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           double, double,
           double*, double*, double*, double*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction_d(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, bool equal_extents,
                                                                       bool lower_extents, bool lower_idx,
                                                                       bool negative_str);

std::tuple<int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           std::complex<float>, std::complex<float>,
           std::complex<float>*, std::complex<float>*, std::complex<float>*, std::complex<float>*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction_c(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, bool equal_extents,
                                                                       bool lower_extents, bool lower_idx,
                                                                       bool negative_str);

std::tuple<int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           std::complex<double>, std::complex<double>,
           std::complex<double>*, std::complex<double>*, std::complex<double>*, std::complex<double>*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction_z(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, bool equal_extents,
                                                                       bool lower_extents, bool lower_idx,
                                                                       bool negative_str);

std::string str(bool b);

int randi(int min, int max);

float randf(float min, float max);

double randd(double min, double max);

std::complex<float> randc(std::complex<float> min, std::complex<float> max);

std::complex<double> randz(std::complex<double> min, std::complex<double> max);

float randf();

double randd();

std::complex<float> randc();

std::complex<double> randz();

char* swap_indices(char* indices, int nmode_A, int nmode_B, int nmode_D);

void print_tensor_s(int nmode, int64_t* extents, int64_t* strides, float* data);

void print_tensor_d(int nmode, int64_t* extents, int64_t* strides, double* data);

void print_tensor_c(int nmode, int64_t* extents, int64_t* strides, std::complex<float>* data);

void print_tensor_z(int nmode, int64_t* extents, int64_t* strides, std::complex<double>* data);

std::tuple<float*, float*> copy_tensor_data_s(int64_t size, float* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str);

std::tuple<double*, double*> copy_tensor_data_d(int64_t size, double* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str);

std::tuple<std::complex<float>*, std::complex<float>*> copy_tensor_data_c(int64_t size, std::complex<float>* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str);

std::tuple<std::complex<double>*, std::complex<double>*> copy_tensor_data_z(int64_t size, std::complex<double>* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str);

float* copy_tensor_data_s(int size, float* data);

void add_incorrect_idx(int64_t max_idx, int* nmode, int64_t** idx, int64_t** extents, int64_t** strides);

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
bool test_contraction_double_precision();
bool test_contraction_complex();
bool test_contraction_complex_double_precision();
bool test_zero_stride();
bool test_error_too_many_idx();
bool test_error_repeated_idx();
bool test_error_non_matching_ext();
bool test_error_C_other_structure();
bool test_error_non_hadamard_shared_idx();
bool test_error_aliasing_within_D();

int main(int argc, char const *argv[])
{
    srand(time(NULL));
    std::cout << "Hadamard Product: " << str(test_hadamard_product()) << std::endl;
    std::cout << "Contration: " << str(test_contraction()) << std::endl;
    std::cout << "Commutativity: " << str(test_commutativity()) << std::endl;
    std::cout << "Permutations: " << str(test_permutations()) << std::endl;
    std::cout << "Equal Extents: " << str(test_equal_extents()) << std::endl;
    std::cout << "Outer Product: " << str(test_outer_product()) << std::endl;
    std::cout << "Full Contraction: " << str(test_full_contraction()) << std::endl;
    std::cout << "Zero Dim Tensor Contraction: " << str(test_zero_dim_tensor_contraction()) << std::endl;
    std::cout << "One Dim Tensor Contraction: " << str(test_one_dim_tensor_contraction()) << std::endl;
    std::cout << "Subtensor Same Index: " << str(test_subtensor_same_idx()) << std::endl;
    std::cout << "Subtensor Lower Index: " << str(test_subtensor_lower_idx()) << std::endl;
    std::cout << "Negative Strides: " << str(test_negative_strides()) << std::endl;
    std::cout << "Negative Strides Subtensor Same Index: " << str(test_negative_strides_subtensor_same_idx()) << std::endl;
    std::cout << "Negative Strides Subtensor Lower Index: " << str(test_negative_strides_subtensor_lower_idx()) << std::endl;
    std::cout << "Contraction Double Precision: " << str(test_contraction_double_precision()) << std::endl;
    std::cout << "Contraction Complex: " << str(test_contraction_complex()) << std::endl;
    std::cout << "Contraction Complex Double Precision: " << str(test_contraction_complex_double_precision()) << std::endl;
    std::cout << "Zero stride: " << str(test_zero_stride()) << std::endl;
    std::cout << "Error: Too Many Indices: " << str(test_error_too_many_idx()) << std::endl;
    std::cout << "Error: Repeated Indices: " << str(test_error_repeated_idx()) << std::endl;
    std::cout << "Error: Non Matching Extents: " << str(test_error_non_matching_ext()) << std::endl;
    std::cout << "Error: C Other Structure: " << str(test_error_C_other_structure()) << std::endl;
    std::cout << "Error: Non Hadamard Shared Indices: " << str(test_error_non_hadamard_shared_idx()) << std::endl;
    std::cout << "Error: Non Aliasing Within D: " << str(test_error_aliasing_within_D()) << std::endl;
    return 0;
}

void run_tblis_mult_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, float* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, float* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, float* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, float* D, int op_D, int64_t* idx_D,
                    float alpha, float beta) {
    tblis::tblis_tensor tblis_A;
    tblis::len_type len_A[nmode_A];
    tblis::stride_type stride_A[nmode_A];
    for (int i = 0; i < nmode_A; i++)
    {
        len_A[i] = extents_A[i];
        stride_A[i] = strides_A[i];
    }

    tblis::tblis_tensor tblis_B;
    tblis::len_type len_B[nmode_B];
    tblis::stride_type stride_B[nmode_B];
    for (int i = 0; i < nmode_B; i++)
    {
        len_B[i] = extents_B[i];
        stride_B[i] = strides_B[i];
    }

    tblis::tblis_tensor tblis_C;
    tblis::len_type len_C[nmode_C];
    tblis::stride_type stride_C[nmode_C];
    for (int i = 0; i < nmode_C; i++)
    {
        len_C[i] = extents_C[i];
        stride_C[i] = strides_C[i];
    }

    tblis::tblis_tensor tblis_D;
    tblis::len_type len_D[nmode_D];
    tblis::stride_type stride_D[nmode_D];
    for (int i = 0; i < nmode_D; i++)
    {
        len_D[i] = extents_D[i];
        stride_D[i] = strides_D[i];
    }

    tblis::tblis_init_tensor_s(&tblis_A, nmode_A, len_A, A, stride_A);
    tblis::tblis_init_tensor_s(&tblis_B, nmode_B, len_B, B, stride_B);
    tblis::tblis_init_tensor_scaled_s(&tblis_C, beta, nmode_C, len_C, C, stride_C);
    tblis::tblis_init_tensor_scaled_s(&tblis_D, 0, nmode_D, len_D, D, stride_D);

    tblis::label_type indices_A[nmode_A + 1];
    tblis::label_type indices_B[nmode_B + 1];
    tblis::label_type indices_C[nmode_C + 1];
    tblis::label_type indices_D[nmode_D + 1];
    indices_A[nmode_A] = '\0';
    indices_B[nmode_B] = '\0';
    indices_C[nmode_C] = '\0';
    indices_D[nmode_D] = '\0';
    for (int i = 0; i < nmode_A; i++)
    {
        indices_A[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        indices_B[i] = idx_B[i];
    }
    for (int i = 0; i < nmode_C; i++)
    {
        indices_C[i] = idx_C[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        indices_D[i] = idx_D[i];
    }

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, indices_D);

    tblis::tblis_tensor_mult(tblis_single, NULL, &tblis_A, indices_A, &tblis_B, indices_B, &tblis_D, indices_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_C, indices_C);

    tblis::tblis_init_tensor_scaled_s(&tblis_D, alpha, nmode_D, len_D, D, stride_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, indices_D);

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, indices_C, &tblis_D, indices_D);
}

void run_tblis_mult_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, double* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, double* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, double* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, double* D, int op_D, int64_t* idx_D,
                    double alpha, double beta) {
    tblis::tblis_tensor tblis_A;
    tblis::len_type len_A[nmode_A];
    tblis::stride_type stride_A[nmode_A];
    for (int i = 0; i < nmode_A; i++)
    {
        len_A[i] = extents_A[i];
        stride_A[i] = strides_A[i];
    }

    tblis::tblis_tensor tblis_B;
    tblis::len_type len_B[nmode_B];
    tblis::stride_type stride_B[nmode_B];
    for (int i = 0; i < nmode_B; i++)
    {
        len_B[i] = extents_B[i];
        stride_B[i] = strides_B[i];
    }

    tblis::tblis_tensor tblis_C;
    tblis::len_type len_C[nmode_C];
    tblis::stride_type stride_C[nmode_C];
    for (int i = 0; i < nmode_C; i++)
    {
        len_C[i] = extents_C[i];
        stride_C[i] = strides_C[i];
    }

    tblis::tblis_tensor tblis_D;
    tblis::len_type len_D[nmode_D];
    tblis::stride_type stride_D[nmode_D];
    for (int i = 0; i < nmode_D; i++)
    {
        len_D[i] = extents_D[i];
        stride_D[i] = strides_D[i];
    }

    tblis::tblis_init_tensor_d(&tblis_A, nmode_A, len_A, A, stride_A);
    tblis::tblis_init_tensor_d(&tblis_B, nmode_B, len_B, B, stride_B);
    tblis::tblis_init_tensor_scaled_d(&tblis_C, beta, nmode_C, len_C, C, stride_C);
    tblis::tblis_init_tensor_scaled_d(&tblis_D, 0, nmode_D, len_D, D, stride_D);

    tblis::label_type indices_A[nmode_A + 1];
    tblis::label_type indices_B[nmode_B + 1];
    tblis::label_type indices_C[nmode_C + 1];
    tblis::label_type indices_D[nmode_D + 1];
    indices_A[nmode_A] = '\0';
    indices_B[nmode_B] = '\0';
    indices_C[nmode_C] = '\0';
    indices_D[nmode_D] = '\0';
    for (int i = 0; i < nmode_A; i++)
    {
        indices_A[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        indices_B[i] = idx_B[i];
    }
    for (int i = 0; i < nmode_C; i++)
    {
        indices_C[i] = idx_C[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        indices_D[i] = idx_D[i];
    }

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, indices_D);

    tblis::tblis_tensor_mult(tblis_single, NULL, &tblis_A, indices_A, &tblis_B, indices_B, &tblis_D, indices_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_C, indices_C);

    tblis::tblis_init_tensor_scaled_d(&tblis_D, alpha, nmode_D, len_D, D, stride_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, indices_D);

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, indices_C, &tblis_D, indices_D);
}

void run_tblis_mult_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<float>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<float>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<float>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<float>* D, int op_D, int64_t* idx_D,
                    std::complex<float> alpha, std::complex<float> beta) {
    tblis::tblis_tensor tblis_A;
    tblis::len_type len_A[nmode_A];
    tblis::stride_type stride_A[nmode_A];
    for (int i = 0; i < nmode_A; i++)
    {
        len_A[i] = extents_A[i];
        stride_A[i] = strides_A[i];
    }

    tblis::tblis_tensor tblis_B;
    tblis::len_type len_B[nmode_B];
    tblis::stride_type stride_B[nmode_B];
    for (int i = 0; i < nmode_B; i++)
    {
        len_B[i] = extents_B[i];
        stride_B[i] = strides_B[i];
    }

    tblis::tblis_tensor tblis_C;
    tblis::len_type len_C[nmode_C];
    tblis::stride_type stride_C[nmode_C];
    for (int i = 0; i < nmode_C; i++)
    {
        len_C[i] = extents_C[i];
        stride_C[i] = strides_C[i];
    }

    tblis::tblis_tensor tblis_D;
    tblis::len_type len_D[nmode_D];
    tblis::stride_type stride_D[nmode_D];
    for (int i = 0; i < nmode_D; i++)
    {
        len_D[i] = extents_D[i];
        stride_D[i] = strides_D[i];
    }

    tblis::label_type indices_A[nmode_A + 1];
    tblis::label_type indices_B[nmode_B + 1];
    tblis::label_type indices_C[nmode_C + 1];
    tblis::label_type indices_D[nmode_D + 1];
    indices_A[nmode_A] = '\0';
    indices_B[nmode_B] = '\0';
    indices_C[nmode_C] = '\0';
    indices_D[nmode_D] = '\0';
    for (int i = 0; i < nmode_A; i++)
    {
        indices_A[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        indices_B[i] = idx_B[i];
    }
    for (int i = 0; i < nmode_C; i++)
    {
        indices_C[i] = idx_C[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        indices_D[i] = idx_D[i];
    }

    tblis::tblis_init_tensor_c(&tblis_A, nmode_A, len_A, A, stride_A);
    tblis::tblis_init_tensor_c(&tblis_B, nmode_B, len_B, B, stride_B);
    tblis::tblis_init_tensor_scaled_c(&tblis_C, beta, nmode_C, len_C, C, stride_C);
    tblis::tblis_init_tensor_scaled_c(&tblis_D, 0, nmode_D, len_D, D, stride_D);

    if (op_A == 1)
    {
        conjugate_c(nmode_A, len_A, stride_A, A);
    }
    if (op_B == 1)
    {
        conjugate_c(nmode_B, len_B, stride_B, B);
    }
    if (op_C == 1)
    {
        conjugate_c(nmode_C, len_C, stride_C, C);
    }

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, indices_D);

    tblis::tblis_tensor_mult(tblis_single, NULL, &tblis_A, indices_A, &tblis_B, indices_B, &tblis_D, indices_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_C, indices_C);

    tblis::tblis_init_tensor_scaled_c(&tblis_D, alpha, nmode_D, len_D, D, stride_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, indices_D);

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, indices_C, &tblis_D, indices_D);

    if (op_D == 1)
    {
        conjugate_c(nmode_D, len_D, stride_D, D);
    }
}

void run_tblis_mult_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<double>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<double>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<double>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<double>* D, int op_D, int64_t* idx_D,
                    std::complex<double> alpha, std::complex<double> beta) {
    tblis::tblis_tensor tblis_A;
    tblis::len_type len_A[nmode_A];
    tblis::stride_type stride_A[nmode_A];
    for (int i = 0; i < nmode_A; i++)
    {
        len_A[i] = extents_A[i];
        stride_A[i] = strides_A[i];
    }

    tblis::tblis_tensor tblis_B;
    tblis::len_type len_B[nmode_B];
    tblis::stride_type stride_B[nmode_B];
    for (int i = 0; i < nmode_B; i++)
    {
        len_B[i] = extents_B[i];
        stride_B[i] = strides_B[i];
    }

    tblis::tblis_tensor tblis_C;
    tblis::len_type len_C[nmode_C];
    tblis::stride_type stride_C[nmode_C];
    for (int i = 0; i < nmode_C; i++)
    {
        len_C[i] = extents_C[i];
        stride_C[i] = strides_C[i];
    }

    tblis::tblis_tensor tblis_D;
    tblis::len_type len_D[nmode_D];
    tblis::stride_type stride_D[nmode_D];
    for (int i = 0; i < nmode_D; i++)
    {
        len_D[i] = extents_D[i];
        stride_D[i] = strides_D[i];
    }

    tblis::label_type indices_A[nmode_A + 1];
    tblis::label_type indices_B[nmode_B + 1];
    tblis::label_type indices_C[nmode_C + 1];
    tblis::label_type indices_D[nmode_D + 1];
    indices_A[nmode_A] = '\0';
    indices_B[nmode_B] = '\0';
    indices_C[nmode_C] = '\0';
    indices_D[nmode_D] = '\0';
    for (int i = 0; i < nmode_A; i++)
    {
        indices_A[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        indices_B[i] = idx_B[i];
    }
    for (int i = 0; i < nmode_C; i++)
    {
        indices_C[i] = idx_C[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        indices_D[i] = idx_D[i];
    }

    tblis::tblis_init_tensor_z(&tblis_A, nmode_A, len_A, A, stride_A);
    tblis::tblis_init_tensor_z(&tblis_B, nmode_B, len_B, B, stride_B);
    tblis::tblis_init_tensor_scaled_z(&tblis_C, beta, nmode_C, len_C, C, stride_C);
    tblis::tblis_init_tensor_scaled_z(&tblis_D, 0, nmode_D, len_D, D, stride_D);

    if (op_A == 1)
    {
        conjugate_z(nmode_A, extents_A, strides_A, A);
    }
    if (op_B == 1)
    {
        conjugate_z(nmode_B, extents_B, strides_B, B);
    }
    if (op_C == 1)
    {
        conjugate_z(nmode_C, extents_C, strides_C, C);
    }

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, indices_D);

    tblis::tblis_tensor_mult(tblis_single, NULL, &tblis_A, indices_A, &tblis_B, indices_B, &tblis_D, indices_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_C, indices_C);

    tblis::tblis_init_tensor_scaled_z(&tblis_D, alpha, nmode_D, len_D, D, stride_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, indices_D);

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, indices_C, &tblis_D, indices_D);

    if (op_D == 1)
    {
        conjugate_z(nmode_D, extents_D, strides_D, D);
    }
}

void conjugate_c(int nmode, int64_t* extents, int64_t* strides, std::complex<float>* T) {
    int64_t coords[nmode];
    int64_t size = 1;
    for (int i = 0; i < nmode; i++)
    {
        coords[i] = 0;
        size *= extents[i];
    }
    for (int i = 0; i < size; i++)
    {
        int64_t index = 0;
        for (int i = 0; i < nmode; i++)
        {
            index += coords[i]*strides[i];
        }
        T[index] = std::conj(T[index]);
        increment_coordinates(coords, nmode, extents);
    }
}

void conjugate_z(int nmode, int64_t* extents, int64_t* strides, std::complex<double>* T) {
    int64_t coords[nmode];
    int64_t size = 1;
    for (int i = 0; i < nmode; i++)
    {
        coords[i] = 0;
        size *= extents[i];
    }
    for (int i = 0; i < size; i++)
    {
        int64_t index = 0;
        for (int i = 0; i < nmode; i++)
        {
            index += coords[i]*strides[i];
        }
        T[index] = std::conj(T[index]);
        increment_coordinates(coords, nmode, extents);
    }
}

bool compare_tensors_s(float* A, float* B, int size) {
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        float rel_diff = abs((A[i] - B[i]) / (A[i] > B[i] ? A[i] : B[i]));
        if (rel_diff > 0.00005)
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << rel_diff << std::endl;
            found = true;
        }
    }
    return !found;
}

bool compare_tensors_d(double* A, double* B, int size) {
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        double rel_diff = abs((A[i] - B[i]) / (A[i] > B[i] ? A[i] : B[i]));
        if (rel_diff > 0.00005)
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << rel_diff << std::endl;
            found = true;
        }
    }
    return !found;
}

bool compare_tensors_c(std::complex<float>* A, std::complex<float>* B, int size) {
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        float rel_diff_r = abs((A[i].real() - B[i].real()) / (A[i].real() > B[i].real() ? A[i].real() : B[i].real()));
        float rel_diff_i = abs((A[i].imag() - B[i].imag()) / (A[i].imag() > B[i].imag() ? A[i].imag() : B[i].imag()));
        if (rel_diff_r > 0.00005 || rel_diff_i > 0.00005)
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << std::complex<float>(rel_diff_r, rel_diff_i) << std::endl;
            found = true;
        }
    }
    return !found;
}

bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size) {
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        double rel_diff_r = abs((A[i].real() - B[i].real()) / (A[i].real() > B[i].real() ? A[i].real() : B[i].real()));
        double rel_diff_i = abs((A[i].imag() - B[i].imag()) / (A[i].imag() > B[i].imag() ? A[i].imag() : B[i].imag()));
        if (rel_diff_r > 0.00005 || rel_diff_i > 0.00005)
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << std::complex<double>(rel_diff_r, rel_diff_i) << std::endl;
            found = true;
        }
    }
    return !found;
}

std::tuple<int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           int, int64_t*, int64_t*, float*, int64_t*,
           float, float,
           float*, float*, float*, float*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction_s(int nmode_A = -1, int nmode_B = -1,
                                                        int nmode_D = randi(0, 4), int contractions = randi(0, 4),
                                                        bool equal_extents = false, bool lower_extents = false,
                                                        bool lower_nmode = false, bool negative_str = false) {
    if (nmode_A == -1 && nmode_B == -1)
    {
        nmode_A = randi(0, nmode_D);
        nmode_B = nmode_D - nmode_A;
        nmode_A = nmode_A + contractions;
        nmode_B = nmode_B + contractions;
    }
    else if (nmode_A == -1)
    {
        contractions = contractions > nmode_B ? randi(0, nmode_B) : contractions;
        nmode_D = nmode_D < nmode_B - contractions ? nmode_B - contractions + randi(0, 4) : nmode_D;
        nmode_A = contractions*2 + nmode_D - nmode_B;
    }
    else if (nmode_B == -1)
    {
        contractions = contractions > nmode_A ? randi(0, nmode_A) : contractions;
        nmode_D = nmode_D < nmode_A - contractions ? nmode_A - contractions + randi(0, 4) : nmode_D;
        nmode_B = contractions*2 + nmode_D - nmode_A;
    }
    else
    {
        contractions = contractions > std::min(nmode_A, nmode_B) ? randi(0, std::min(nmode_A, nmode_B)) : contractions;
        nmode_D = nmode_A + nmode_B - contractions * 2;
    }
    int nmode_C = nmode_D;    

    int64_t* idx_A = new int64_t[nmode_A];
    for (int i = 0; i < nmode_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    if (nmode_A > 0) {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }
    
    int64_t* idx_B = new int64_t[nmode_B];
    int idx_contracted[contractions];
    for (int i = 0; i < contractions; i++)
    {
        idx_B[i] = idx_A[i];
        idx_contracted[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B - contractions; i++)
    {
        idx_B[i + contractions] = 'a' + nmode_A + i;
    }
    if (nmode_B > 0) {
        std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    }
    if (nmode_A > 0) {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }

    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    int index = 0;
    for (int j = 0; j < nmode_A + nmode_B - contractions; j++)
    {
        int64_t idx = 'a' + j;
        bool found = false;
        for (int i = 0; i < contractions; i++)
        {
            if (idx == idx_contracted[i])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            idx_D[index] = idx;
            index++;
        }
    }
    if (nmode_D > 0) {
        std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    }
    std::copy(idx_D, idx_D + nmode_D, idx_C);

    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    int64_t extent = randi(1, 4);
    for (int i = 0; i < nmode_A; i++)
    {
        extents_A[i] = equal_extents ? randi(1, 4) : extent;
    }
    for (int i = 0; i < nmode_B; i++)
    {
        int found = -1;
        for (int j = 0; j < nmode_A; j++)
        {
            if (idx_B[i] == idx_A[j])
            {
                found = j;
                break;
            }
        }
        if (found != -1)
        {
            extents_B[i] = extents_A[found];
        }
        else
        {
            extents_B[i] = equal_extents ? randi(1, 4) : extent;
        }
    }
    for (int i = 0; i < nmode_D; i++)
    {
        int found_A = -1;
        for (int j = 0; j < nmode_A; j++)
        {
            if (idx_D[i] == idx_A[j])
            {
                found_A = j;
                break;
            }
        }

        int found_B = -1;
        for (int j = 0; j < nmode_B; j++)
        {
            if (idx_D[i] == idx_B[j])
            {
                found_B = j;
                break;
            }
        }

        if (found_A != -1)
        {
            extents_D[i] = extents_A[found_A];
        }
        else if (found_B != -1)
        {
            extents_D[i] = extents_B[found_B];
        }
        else
        {
            std::cout << "Error: Index not found" << std::endl;
        }
    }    
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);

    int outer_nmode_A = lower_nmode ? nmode_A + randi(1, 4) : nmode_A;
    int outer_nmode_B = lower_nmode ? nmode_B + randi(1, 4) : nmode_B;
    int outer_nmode_C = lower_nmode ? nmode_C + randi(1, 4) : nmode_C;
    int outer_nmode_D = lower_nmode ? nmode_D + randi(1, 4) : nmode_D;
    int64_t outer_extents_A[outer_nmode_A];
    int64_t outer_extents_B[outer_nmode_B];
    int64_t outer_extents_C[outer_nmode_C];
    int64_t outer_extents_D[outer_nmode_D];
    int64_t* strides_A = new int64_t[nmode_A];
    int64_t* strides_B = new int64_t[nmode_B];
    int64_t* strides_C = new int64_t[nmode_C];
    int64_t* strides_D = new int64_t[nmode_D];
    int64_t* offset_A = new int64_t[nmode_A];
    int64_t* offset_B = new int64_t[nmode_B];
    int64_t* offset_C = new int64_t[nmode_C];
    int64_t* offset_D = new int64_t[nmode_D];
    int64_t size_A = 1;
    int64_t size_B = 1;
    int64_t size_C = 1;
    int64_t size_D = 1;

    int64_t str = negative_str ? -1 : 1;
    int idx = 0;
    for (int i = 0; i < outer_nmode_A; i++)
    {
        if ((randf(0, 1) < (float)nmode_A/(float)outer_nmode_A || outer_nmode_A - i == nmode_A - idx) && nmode_A - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_A[i] = lower_extents ? extents_A[idx] + extension : extents_A[idx];
            offset_A[idx] = lower_extents && outer_extents_A[i] - extents_A[idx] > 0 ? randi(0, outer_extents_A[i] - extents_A[idx]) : 0;
            strides_A[idx] = str;
            str *= outer_extents_A[i];
            idx++;
        }
        else
        {
            outer_extents_A[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_A[i];
        }
        size_A *= outer_extents_A[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_B; i++)
    {
        if ((randf(0, 1) < (float)nmode_B/(float)outer_nmode_B || outer_nmode_B - i == nmode_B - idx) && nmode_B - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_B[i] = lower_extents ? extents_B[idx] + extension : extents_B[idx];
            offset_B[idx] = lower_extents && outer_extents_B[i] - extents_B[idx] > 0 ? randi(0, outer_extents_B[i] - extents_B[idx]) : 0;
            strides_B[idx] = str;
            str *= outer_extents_B[i];
            idx++;
        }
        else
        {
            outer_extents_B[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_B[i];
        }
        size_B *= outer_extents_B[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_C; i++)
    {
        if ((randf(0, 1) < (float)nmode_C/(float)outer_nmode_C || outer_nmode_C - i == nmode_C - idx) && nmode_C - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_C[i] = lower_extents ? extents_C[idx] + extension : extents_C[idx];
            offset_C[idx] = lower_extents && outer_extents_C[i] - extents_C[idx] > 0 ? randi(0, outer_extents_C[i] - extents_C[idx]) : 0;
            strides_C[idx] = str;
            str *= outer_extents_C[i];
            idx++;
        }
        else
        {
            outer_extents_C[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_C[i];
        }
        size_C *= outer_extents_C[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_D; i++)
    {
        if ((randf(0, 1) < (float)nmode_D/(float)outer_nmode_D || outer_nmode_D - i == nmode_D - idx) && nmode_D - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_D[i] = lower_extents ? extents_D[idx] + extension : extents_D[idx];
            offset_D[idx] = lower_extents && outer_extents_D[i] - extents_D[idx] > 0 ? randi(0, outer_extents_D[i] - extents_D[idx]) : 0;
            strides_D[idx] = str;
            str *= outer_extents_D[i];
            idx++;
        }
        else
        {
            outer_extents_D[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_D[i];
        }
        size_D *= outer_extents_D[i];
    }

    float* data_A = new float[size_A];
    float* data_B = new float[size_B];
    float* data_C = new float[size_C];
    float* data_D = new float[size_D];

    for (int i = 0; i < size_A; i++)
    {
        data_A[i] = randf();
    }
    for (int i = 0; i < size_B; i++)
    {
        data_B[i] = randf();
    }
    for (int i = 0; i < size_C; i++)
    {
        data_C[i] = randf();
    }
    for (int i = 0; i < size_D; i++)
    {
        data_D[i] = randf();
    }

    float* A = negative_str ? data_A + size_A - 1 : data_A;
    float* B = negative_str ? data_B + size_B - 1 : data_B;
    float* C = negative_str ? data_C + size_C - 1 : data_C;
    float* D = negative_str ? data_D + size_D - 1 : data_D;

    for (int i = 0; i < nmode_A; i++)
    {
        A += offset_A[i] * strides_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        B += offset_B[i] * strides_B[i];
    }
    for (int i = 0; i < nmode_C; i++)
    {
        C += offset_C[i] * strides_C[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        D += offset_D[i] * strides_D[i];
    }

    float alpha = randf();
    float beta = randf();

    return {nmode_A, extents_A, strides_A, A, idx_A,
            nmode_B, extents_B, strides_B, B, idx_B,
            nmode_C, extents_C, strides_C, C, idx_C,
            nmode_D, extents_D, strides_D, D, idx_D,
            alpha, beta,
            data_A, data_B, data_C, data_D,
            size_A, size_B, size_C, size_D,
            offset_A, offset_B, offset_C, offset_D};
}

std::tuple<int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           double, double,
           double*, double*, double*, double*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction_d(int nmode_A = -1, int nmode_B = -1,
                                                        int nmode_D = randi(0, 4), int contractions = randi(0, 4),
                                                        bool equal_extents = false, bool lower_extents = false,
                                                        bool lower_nmode = false, bool negative_str = false) {
    if (nmode_A == -1 && nmode_B == -1)
    {
        nmode_A = randi(0, nmode_D);
        nmode_B = nmode_D - nmode_A;
        nmode_A = nmode_A + contractions;
        nmode_B = nmode_B + contractions;
    }
    else if (nmode_A == -1)
    {
        contractions = contractions > nmode_B ? randi(0, nmode_B) : contractions;
        nmode_D = nmode_D < nmode_B - contractions ? nmode_B - contractions + randi(0, 4) : nmode_D;
        nmode_A = contractions*2 + nmode_D - nmode_B;
    }
    else if (nmode_B == -1)
    {
        contractions = contractions > nmode_A ? randi(0, nmode_A) : contractions;
        nmode_D = nmode_D < nmode_A - contractions ? nmode_A - contractions + randi(0, 4) : nmode_D;
        nmode_B = contractions*2 + nmode_D - nmode_A;
    }
    else
    {
        contractions = contractions > std::min(nmode_A, nmode_B) ? randi(0, std::min(nmode_A, nmode_B)) : contractions;
        nmode_D = nmode_A + nmode_B - contractions * 2;
    }
    int nmode_C = nmode_D;    

    int64_t* idx_A = new int64_t[nmode_A];
    for (int i = 0; i < nmode_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    if (nmode_A > 0) {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }
    
    int64_t* idx_B = new int64_t[nmode_B];
    int idx_contracted[contractions];
    for (int i = 0; i < contractions; i++)
    {
        idx_B[i] = idx_A[i];
        idx_contracted[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B - contractions; i++)
    {
        idx_B[i + contractions] = 'a' + nmode_A + i;
    }
    if (nmode_B > 0) {
        std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    }
    if (nmode_A > 0) {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }

    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    int index = 0;
    for (int j = 0; j < nmode_A + nmode_B - contractions; j++)
    {
        int64_t idx = 'a' + j;
        bool found = false;
        for (int i = 0; i < contractions; i++)
        {
            if (idx == idx_contracted[i])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            idx_D[index] = idx;
            index++;
        }
    }
    if (nmode_D > 0) {
        std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    }
    std::copy(idx_D, idx_D + nmode_D, idx_C);

    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    int64_t extent = randi(1, 4);
    for (int i = 0; i < nmode_A; i++)
    {
        extents_A[i] = equal_extents ? randi(1, 4) : extent;
    }
    for (int i = 0; i < nmode_B; i++)
    {
        int found = -1;
        for (int j = 0; j < nmode_A; j++)
        {
            if (idx_B[i] == idx_A[j])
            {
                found = j;
                break;
            }
        }
        if (found != -1)
        {
            extents_B[i] = extents_A[found];
        }
        else
        {
            extents_B[i] = equal_extents ? randi(1, 4) : extent;
        }
    }
    for (int i = 0; i < nmode_D; i++)
    {
        int found_A = -1;
        for (int j = 0; j < nmode_A; j++)
        {
            if (idx_D[i] == idx_A[j])
            {
                found_A = j;
                break;
            }
        }

        int found_B = -1;
        for (int j = 0; j < nmode_B; j++)
        {
            if (idx_D[i] == idx_B[j])
            {
                found_B = j;
                break;
            }
        }

        if (found_A != -1)
        {
            extents_D[i] = extents_A[found_A];
        }
        else if (found_B != -1)
        {
            extents_D[i] = extents_B[found_B];
        }
        else
        {
            std::cout << "Error: Index not found" << std::endl;
        }
    }    
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);

    int outer_nmode_A = lower_nmode ? nmode_A + randi(1, 4) : nmode_A;
    int outer_nmode_B = lower_nmode ? nmode_B + randi(1, 4) : nmode_B;
    int outer_nmode_C = lower_nmode ? nmode_C + randi(1, 4) : nmode_C;
    int outer_nmode_D = lower_nmode ? nmode_D + randi(1, 4) : nmode_D;
    int64_t outer_extents_A[outer_nmode_A];
    int64_t outer_extents_B[outer_nmode_B];
    int64_t outer_extents_C[outer_nmode_C];
    int64_t outer_extents_D[outer_nmode_D];
    int64_t* strides_A = new int64_t[nmode_A];
    int64_t* strides_B = new int64_t[nmode_B];
    int64_t* strides_C = new int64_t[nmode_C];
    int64_t* strides_D = new int64_t[nmode_D];
    int64_t* offset_A = new int64_t[nmode_A];
    int64_t* offset_B = new int64_t[nmode_B];
    int64_t* offset_C = new int64_t[nmode_C];
    int64_t* offset_D = new int64_t[nmode_D];
    int64_t size_A = 1;
    int64_t size_B = 1;
    int64_t size_C = 1;
    int64_t size_D = 1;

    int64_t str = negative_str ? -1 : 1;
    int idx = 0;
    for (int i = 0; i < outer_nmode_A; i++)
    {
        if ((randf(0, 1) < (float)nmode_A/(float)outer_nmode_A || outer_nmode_A - i == nmode_A - idx) && nmode_A - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_A[i] = lower_extents ? extents_A[idx] + extension : extents_A[idx];
            offset_A[idx] = lower_extents && outer_extents_A[i] - extents_A[idx] > 0 ? randi(0, outer_extents_A[i] - extents_A[idx]) : 0;
            strides_A[idx] = str;
            str *= outer_extents_A[i];
            idx++;
        }
        else
        {
            outer_extents_A[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_A[i];
        }
        size_A *= outer_extents_A[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_B; i++)
    {
        if ((randf(0, 1) < (float)nmode_B/(float)outer_nmode_B || outer_nmode_B - i == nmode_B - idx) && nmode_B - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_B[i] = lower_extents ? extents_B[idx] + extension : extents_B[idx];
            offset_B[idx] = lower_extents && outer_extents_B[i] - extents_B[idx] > 0 ? randi(0, outer_extents_B[i] - extents_B[idx]) : 0;
            strides_B[idx] = str;
            str *= outer_extents_B[i];
            idx++;
        }
        else
        {
            outer_extents_B[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_B[i];
        }
        size_B *= outer_extents_B[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_C; i++)
    {
        if ((randf(0, 1) < (float)nmode_C/(float)outer_nmode_C || outer_nmode_C - i == nmode_C - idx) && nmode_C - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_C[i] = lower_extents ? extents_C[idx] + extension : extents_C[idx];
            offset_C[idx] = lower_extents && outer_extents_C[i] - extents_C[idx] > 0 ? randi(0, outer_extents_C[i] - extents_C[idx]) : 0;
            strides_C[idx] = str;
            str *= outer_extents_C[i];
            idx++;
        }
        else
        {
            outer_extents_C[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_C[i];
        }
        size_C *= outer_extents_C[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_D; i++)
    {
        if ((randf(0, 1) < (float)nmode_D/(float)outer_nmode_D || outer_nmode_D - i == nmode_D - idx) && nmode_D - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_D[i] = lower_extents ? extents_D[idx] + extension : extents_D[idx];
            offset_D[idx] = lower_extents && outer_extents_D[i] - extents_D[idx] > 0 ? randi(0, outer_extents_D[i] - extents_D[idx]) : 0;
            strides_D[idx] = str;
            str *= outer_extents_D[i];
            idx++;
        }
        else
        {
            outer_extents_D[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_D[i];
        }
        size_D *= outer_extents_D[i];
    }

    double* data_A = new double[size_A];
    double* data_B = new double[size_B];
    double* data_C = new double[size_C];
    double* data_D = new double[size_D];

    for (int i = 0; i < size_A; i++)
    {
        data_A[i] = randd();
    }
    for (int i = 0; i < size_B; i++)
    {
        data_B[i] = randd();
    }
    for (int i = 0; i < size_C; i++)
    {
        data_C[i] = randd();
    }
    for (int i = 0; i < size_D; i++)
    {
        data_D[i] = randd();
    }

    double* A = negative_str ? data_A + size_A - 1 : data_A;
    double* B = negative_str ? data_B + size_B - 1 : data_B;
    double* C = negative_str ? data_C + size_C - 1 : data_C;
    double* D = negative_str ? data_D + size_D - 1 : data_D;

    for (int i = 0; i < nmode_A; i++)
    {
        A += offset_A[i] * strides_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        B += offset_B[i] * strides_B[i];
    }
    for (int i = 0; i < nmode_C; i++)
    {
        C += offset_C[i] * strides_C[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        D += offset_D[i] * strides_D[i];
    }

    double alpha = randd();
    double beta = randd();

    return {nmode_A, extents_A, strides_A, A, idx_A,
            nmode_B, extents_B, strides_B, B, idx_B,
            nmode_C, extents_C, strides_C, C, idx_C,
            nmode_D, extents_D, strides_D, D, idx_D,
            alpha, beta,
            data_A, data_B, data_C, data_D,
            size_A, size_B, size_C, size_D,
            offset_A, offset_B, offset_C, offset_D};
}

std::tuple<int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           std::complex<float>, std::complex<float>,
           std::complex<float>*, std::complex<float>*, std::complex<float>*, std::complex<float>*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction_c(int nmode_A = -1, int nmode_B = -1,
                                                        int nmode_D = randi(0, 4), int contractions = randi(0, 4),
                                                        bool equal_extents = false, bool lower_extents = false,
                                                        bool lower_nmode = false, bool negative_str = false) {
    if (nmode_A == -1 && nmode_B == -1)
    {
        nmode_A = randi(0, nmode_D);
        nmode_B = nmode_D - nmode_A;
        nmode_A = nmode_A + contractions;
        nmode_B = nmode_B + contractions;
    }
    else if (nmode_A == -1)
    {
        contractions = contractions > nmode_B ? randi(0, nmode_B) : contractions;
        nmode_D = nmode_D < nmode_B - contractions ? nmode_B - contractions + randi(0, 4) : nmode_D;
        nmode_A = contractions*2 + nmode_D - nmode_B;
    }
    else if (nmode_B == -1)
    {
        contractions = contractions > nmode_A ? randi(0, nmode_A) : contractions;
        nmode_D = nmode_D < nmode_A - contractions ? nmode_A - contractions + randi(0, 4) : nmode_D;
        nmode_B = contractions*2 + nmode_D - nmode_A;
    }
    else
    {
        contractions = contractions > std::min(nmode_A, nmode_B) ? randi(0, std::min(nmode_A, nmode_B)) : contractions;
        nmode_D = nmode_A + nmode_B - contractions * 2;
    }
    int nmode_C = nmode_D;    

    int64_t* idx_A = new int64_t[nmode_A];
    for (int i = 0; i < nmode_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    if (nmode_A > 0) {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }
    
    int64_t* idx_B = new int64_t[nmode_B];
    int idx_contracted[contractions];
    for (int i = 0; i < contractions; i++)
    {
        idx_B[i] = idx_A[i];
        idx_contracted[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B - contractions; i++)
    {
        idx_B[i + contractions] = 'a' + nmode_A + i;
    }
    if (nmode_B > 0) {
        std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    }
    if (nmode_A > 0) {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }

    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    int index = 0;
    for (int j = 0; j < nmode_A + nmode_B - contractions; j++)
    {
        int64_t idx = 'a' + j;
        bool found = false;
        for (int i = 0; i < contractions; i++)
        {
            if (idx == idx_contracted[i])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            idx_D[index] = idx;
            index++;
        }
    }
    if (nmode_D > 0) {
        std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    }
    std::copy(idx_D, idx_D + nmode_D, idx_C);

    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    int64_t extent = randi(1, 4);
    for (int i = 0; i < nmode_A; i++)
    {
        extents_A[i] = equal_extents ? randi(1, 4) : extent;
    }
    for (int i = 0; i < nmode_B; i++)
    {
        int found = -1;
        for (int j = 0; j < nmode_A; j++)
        {
            if (idx_B[i] == idx_A[j])
            {
                found = j;
                break;
            }
        }
        if (found != -1)
        {
            extents_B[i] = extents_A[found];
        }
        else
        {
            extents_B[i] = equal_extents ? randi(1, 4) : extent;
        }
    }
    for (int i = 0; i < nmode_D; i++)
    {
        int found_A = -1;
        for (int j = 0; j < nmode_A; j++)
        {
            if (idx_D[i] == idx_A[j])
            {
                found_A = j;
                break;
            }
        }

        int found_B = -1;
        for (int j = 0; j < nmode_B; j++)
        {
            if (idx_D[i] == idx_B[j])
            {
                found_B = j;
                break;
            }
        }

        if (found_A != -1)
        {
            extents_D[i] = extents_A[found_A];
        }
        else if (found_B != -1)
        {
            extents_D[i] = extents_B[found_B];
        }
        else
        {
            std::cout << "Error: Index not found" << std::endl;
        }
    }    
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);

    int outer_nmode_A = lower_nmode ? nmode_A + randi(1, 4) : nmode_A;
    int outer_nmode_B = lower_nmode ? nmode_B + randi(1, 4) : nmode_B;
    int outer_nmode_C = lower_nmode ? nmode_C + randi(1, 4) : nmode_C;
    int outer_nmode_D = lower_nmode ? nmode_D + randi(1, 4) : nmode_D;
    int64_t outer_extents_A[outer_nmode_A];
    int64_t outer_extents_B[outer_nmode_B];
    int64_t outer_extents_C[outer_nmode_C];
    int64_t outer_extents_D[outer_nmode_D];
    int64_t* strides_A = new int64_t[nmode_A];
    int64_t* strides_B = new int64_t[nmode_B];
    int64_t* strides_C = new int64_t[nmode_C];
    int64_t* strides_D = new int64_t[nmode_D];
    int64_t* offset_A = new int64_t[nmode_A];
    int64_t* offset_B = new int64_t[nmode_B];
    int64_t* offset_C = new int64_t[nmode_C];
    int64_t* offset_D = new int64_t[nmode_D];
    int64_t size_A = 1;
    int64_t size_B = 1;
    int64_t size_C = 1;
    int64_t size_D = 1;

    int64_t str = negative_str ? -1 : 1;
    int idx = 0;
    for (int i = 0; i < outer_nmode_A; i++)
    {
        if ((randf(0, 1) < (float)nmode_A/(float)outer_nmode_A || outer_nmode_A - i == nmode_A - idx) && nmode_A - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_A[i] = lower_extents ? extents_A[idx] + extension : extents_A[idx];
            offset_A[idx] = lower_extents && outer_extents_A[i] - extents_A[idx] > 0 ? randi(0, outer_extents_A[i] - extents_A[idx]) : 0;
            strides_A[idx] = str;
            str *= outer_extents_A[i];
            idx++;
        }
        else
        {
            outer_extents_A[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_A[i];
        }
        size_A *= outer_extents_A[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_B; i++)
    {
        if ((randf(0, 1) < (float)nmode_B/(float)outer_nmode_B || outer_nmode_B - i == nmode_B - idx) && nmode_B - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_B[i] = lower_extents ? extents_B[idx] + extension : extents_B[idx];
            offset_B[idx] = lower_extents && outer_extents_B[i] - extents_B[idx] > 0 ? randi(0, outer_extents_B[i] - extents_B[idx]) : 0;
            strides_B[idx] = str;
            str *= outer_extents_B[i];
            idx++;
        }
        else
        {
            outer_extents_B[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_B[i];
        }
        size_B *= outer_extents_B[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_C; i++)
    {
        if ((randf(0, 1) < (float)nmode_C/(float)outer_nmode_C || outer_nmode_C - i == nmode_C - idx) && nmode_C - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_C[i] = lower_extents ? extents_C[idx] + extension : extents_C[idx];
            offset_C[idx] = lower_extents && outer_extents_C[i] - extents_C[idx] > 0 ? randi(0, outer_extents_C[i] - extents_C[idx]) : 0;
            strides_C[idx] = str;
            str *= outer_extents_C[i];
            idx++;
        }
        else
        {
            outer_extents_C[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_C[i];
        }
        size_C *= outer_extents_C[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_D; i++)
    {
        if ((randf(0, 1) < (float)nmode_D/(float)outer_nmode_D || outer_nmode_D - i == nmode_D - idx) && nmode_D - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_D[i] = lower_extents ? extents_D[idx] + extension : extents_D[idx];
            offset_D[idx] = lower_extents && outer_extents_D[i] - extents_D[idx] > 0 ? randi(0, outer_extents_D[i] - extents_D[idx]) : 0;
            strides_D[idx] = str;
            str *= outer_extents_D[i];
            idx++;
        }
        else
        {
            outer_extents_D[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_D[i];
        }
        size_D *= outer_extents_D[i];
    }

    std::complex<float>* data_A = new std::complex<float>[size_A];
    std::complex<float>* data_B = new std::complex<float>[size_B];
    std::complex<float>* data_C = new std::complex<float>[size_C];
    std::complex<float>* data_D = new std::complex<float>[size_D];

    for (int i = 0; i < size_A; i++)
    {
        data_A[i] = randc();
    }
    for (int i = 0; i < size_B; i++)
    {
        data_B[i] = randc();
    }
    for (int i = 0; i < size_C; i++)
    {
        data_C[i] = randc();
    }
    for (int i = 0; i < size_D; i++)
    {
        data_D[i] = randc();
    }

    std::complex<float>* A = negative_str ? data_A + size_A - 1 : data_A;
    std::complex<float>* B = negative_str ? data_B + size_B - 1 : data_B;
    std::complex<float>* C = negative_str ? data_C + size_C - 1 : data_C;
    std::complex<float>* D = negative_str ? data_D + size_D - 1 : data_D;

    for (int i = 0; i < nmode_A; i++)
    {
        A += offset_A[i] * strides_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        B += offset_B[i] * strides_B[i];
    }
    for (int i = 0; i < nmode_C; i++)
    {
        C += offset_C[i] * strides_C[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        D += offset_D[i] * strides_D[i];
    }

    std::complex<float> alpha = randc();
    std::complex<float> beta = randc();

    return {nmode_A, extents_A, strides_A, A, idx_A,
            nmode_B, extents_B, strides_B, B, idx_B,
            nmode_C, extents_C, strides_C, C, idx_C,
            nmode_D, extents_D, strides_D, D, idx_D,
            alpha, beta,
            data_A, data_B, data_C, data_D,
            size_A, size_B, size_C, size_D,
            offset_A, offset_B, offset_C, offset_D};
}

std::tuple<int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           std::complex<double>, std::complex<double>,
           std::complex<double>*, std::complex<double>*, std::complex<double>*, std::complex<double>*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction_z(int nmode_A = -1, int nmode_B = -1,
                                                        int nmode_D = randi(0, 4), int contractions = randi(0, 4),
                                                        bool equal_extents = false, bool lower_extents = false,
                                                        bool lower_nmode = false, bool negative_str = false) {
    if (nmode_A == -1 && nmode_B == -1)
    {
        nmode_A = randi(0, nmode_D);
        nmode_B = nmode_D - nmode_A;
        nmode_A = nmode_A + contractions;
        nmode_B = nmode_B + contractions;
    }
    else if (nmode_A == -1)
    {
        contractions = contractions > nmode_B ? randi(0, nmode_B) : contractions;
        nmode_D = nmode_D < nmode_B - contractions ? nmode_B - contractions + randi(0, 4) : nmode_D;
        nmode_A = contractions*2 + nmode_D - nmode_B;
    }
    else if (nmode_B == -1)
    {
        contractions = contractions > nmode_A ? randi(0, nmode_A) : contractions;
        nmode_D = nmode_D < nmode_A - contractions ? nmode_A - contractions + randi(0, 4) : nmode_D;
        nmode_B = contractions*2 + nmode_D - nmode_A;
    }
    else
    {
        contractions = contractions > std::min(nmode_A, nmode_B) ? randi(0, std::min(nmode_A, nmode_B)) : contractions;
        nmode_D = nmode_A + nmode_B - contractions * 2;
    }
    int nmode_C = nmode_D;    

    int64_t* idx_A = new int64_t[nmode_A];
    for (int i = 0; i < nmode_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    if (nmode_A > 0) {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }
    
    int64_t* idx_B = new int64_t[nmode_B];
    int idx_contracted[contractions];
    for (int i = 0; i < contractions; i++)
    {
        idx_B[i] = idx_A[i];
        idx_contracted[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B - contractions; i++)
    {
        idx_B[i + contractions] = 'a' + nmode_A + i;
    }
    if (nmode_B > 0) {
        std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    }
    if (nmode_A > 0) {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }

    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    int index = 0;
    for (int j = 0; j < nmode_A + nmode_B - contractions; j++)
    {
        int64_t idx = 'a' + j;
        bool found = false;
        for (int i = 0; i < contractions; i++)
        {
            if (idx == idx_contracted[i])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            idx_D[index] = idx;
            index++;
        }
    }
    if (nmode_D > 0) {
        std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    }
    std::copy(idx_D, idx_D + nmode_D, idx_C);

    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    int64_t extent = randi(1, 4);
    for (int i = 0; i < nmode_A; i++)
    {
        extents_A[i] = equal_extents ? randi(1, 4) : extent;
    }
    for (int i = 0; i < nmode_B; i++)
    {
        int found = -1;
        for (int j = 0; j < nmode_A; j++)
        {
            if (idx_B[i] == idx_A[j])
            {
                found = j;
                break;
            }
        }
        if (found != -1)
        {
            extents_B[i] = extents_A[found];
        }
        else
        {
            extents_B[i] = equal_extents ? randi(1, 4) : extent;
        }
    }
    for (int i = 0; i < nmode_D; i++)
    {
        int found_A = -1;
        for (int j = 0; j < nmode_A; j++)
        {
            if (idx_D[i] == idx_A[j])
            {
                found_A = j;
                break;
            }
        }

        int found_B = -1;
        for (int j = 0; j < nmode_B; j++)
        {
            if (idx_D[i] == idx_B[j])
            {
                found_B = j;
                break;
            }
        }

        if (found_A != -1)
        {
            extents_D[i] = extents_A[found_A];
        }
        else if (found_B != -1)
        {
            extents_D[i] = extents_B[found_B];
        }
        else
        {
            std::cout << "Error: Index not found" << std::endl;
        }
    }    
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);

    int outer_nmode_A = lower_nmode ? nmode_A + randi(1, 4) : nmode_A;
    int outer_nmode_B = lower_nmode ? nmode_B + randi(1, 4) : nmode_B;
    int outer_nmode_C = lower_nmode ? nmode_C + randi(1, 4) : nmode_C;
    int outer_nmode_D = lower_nmode ? nmode_D + randi(1, 4) : nmode_D;
    int64_t outer_extents_A[outer_nmode_A];
    int64_t outer_extents_B[outer_nmode_B];
    int64_t outer_extents_C[outer_nmode_C];
    int64_t outer_extents_D[outer_nmode_D];
    int64_t* strides_A = new int64_t[nmode_A];
    int64_t* strides_B = new int64_t[nmode_B];
    int64_t* strides_C = new int64_t[nmode_C];
    int64_t* strides_D = new int64_t[nmode_D];
    int64_t* offset_A = new int64_t[nmode_A];
    int64_t* offset_B = new int64_t[nmode_B];
    int64_t* offset_C = new int64_t[nmode_C];
    int64_t* offset_D = new int64_t[nmode_D];
    int64_t size_A = 1;
    int64_t size_B = 1;
    int64_t size_C = 1;
    int64_t size_D = 1;

    int64_t str = negative_str ? -1 : 1;
    int idx = 0;
    for (int i = 0; i < outer_nmode_A; i++)
    {
        if ((randf(0, 1) < (float)nmode_A/(float)outer_nmode_A || outer_nmode_A - i == nmode_A - idx) && nmode_A - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_A[i] = lower_extents ? extents_A[idx] + extension : extents_A[idx];
            offset_A[idx] = lower_extents && outer_extents_A[i] - extents_A[idx] > 0 ? randi(0, outer_extents_A[i] - extents_A[idx]) : 0;
            strides_A[idx] = str;
            str *= outer_extents_A[i];
            idx++;
        }
        else
        {
            outer_extents_A[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_A[i];
        }
        size_A *= outer_extents_A[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_B; i++)
    {
        if ((randf(0, 1) < (float)nmode_B/(float)outer_nmode_B || outer_nmode_B - i == nmode_B - idx) && nmode_B - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_B[i] = lower_extents ? extents_B[idx] + extension : extents_B[idx];
            offset_B[idx] = lower_extents && outer_extents_B[i] - extents_B[idx] > 0 ? randi(0, outer_extents_B[i] - extents_B[idx]) : 0;
            strides_B[idx] = str;
            str *= outer_extents_B[i];
            idx++;
        }
        else
        {
            outer_extents_B[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_B[i];
        }
        size_B *= outer_extents_B[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_C; i++)
    {
        if ((randf(0, 1) < (float)nmode_C/(float)outer_nmode_C || outer_nmode_C - i == nmode_C - idx) && nmode_C - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_C[i] = lower_extents ? extents_C[idx] + extension : extents_C[idx];
            offset_C[idx] = lower_extents && outer_extents_C[i] - extents_C[idx] > 0 ? randi(0, outer_extents_C[i] - extents_C[idx]) : 0;
            strides_C[idx] = str;
            str *= outer_extents_C[i];
            idx++;
        }
        else
        {
            outer_extents_C[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_C[i];
        }
        size_C *= outer_extents_C[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_nmode_D; i++)
    {
        if ((randf(0, 1) < (float)nmode_D/(float)outer_nmode_D || outer_nmode_D - i == nmode_D - idx) && nmode_D - idx > 0)
        {
            int extension = randi(1, 4);
            outer_extents_D[i] = lower_extents ? extents_D[idx] + extension : extents_D[idx];
            offset_D[idx] = lower_extents && outer_extents_D[i] - extents_D[idx] > 0 ? randi(0, outer_extents_D[i] - extents_D[idx]) : 0;
            strides_D[idx] = str;
            str *= outer_extents_D[i];
            idx++;
        }
        else
        {
            outer_extents_D[i] = lower_extents ? randi(1, 8) : randi(1, 4);
            str *= outer_extents_D[i];
        }
        size_D *= outer_extents_D[i];
    }

    std::complex<double>* data_A = new std::complex<double>[size_A];
    std::complex<double>* data_B = new std::complex<double>[size_B];
    std::complex<double>* data_C = new std::complex<double>[size_C];
    std::complex<double>* data_D = new std::complex<double>[size_D];

    for (int i = 0; i < size_A; i++)
    {
        data_A[i] = randz();
    }
    for (int i = 0; i < size_B; i++)
    {
        data_B[i] = randz();
    }
    for (int i = 0; i < size_C; i++)
    {
        data_C[i] = randz();
    }
    for (int i = 0; i < size_D; i++)
    {
        data_D[i] = randz();
    }

    std::complex<double>* A = negative_str ? data_A + size_A - 1 : data_A;
    std::complex<double>* B = negative_str ? data_B + size_B - 1 : data_B;
    std::complex<double>* C = negative_str ? data_C + size_C - 1 : data_C;
    std::complex<double>* D = negative_str ? data_D + size_D - 1 : data_D;

    for (int i = 0; i < nmode_A; i++)
    {
        A += offset_A[i] * strides_A[i];
    }
    for (int i = 0; i < nmode_B; i++)
    {
        B += offset_B[i] * strides_B[i];
    }
    for (int i = 0; i < nmode_C; i++)
    {
        C += offset_C[i] * strides_C[i];
    }
    for (int i = 0; i < nmode_D; i++)
    {
        D += offset_D[i] * strides_D[i];
    }

    std::complex<double> alpha = randz();
    std::complex<double> beta = randz();

    return {nmode_A, extents_A, strides_A, A, idx_A,
            nmode_B, extents_B, strides_B, B, idx_B,
            nmode_C, extents_C, strides_C, C, idx_C,
            nmode_D, extents_D, strides_D, D, idx_D,
            alpha, beta,
            data_A, data_B, data_C, data_D,
            size_A, size_B, size_C, size_D,
            offset_A, offset_B, offset_C, offset_D};
}

std::tuple<float*, float*> copy_tensor_data_s(int64_t size, float* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str = false) {
    float* dataA = new float[size];
    std::copy(data, data + size, dataA);
    float* A = negative_str ? dataA + size - 1 : dataA;
    for (int i = 0; i < nmode; i++)
    {
        A += offset[i] * strides[i];
    }
    return {A, dataA};
}

std::tuple<double*, double*> copy_tensor_data_d(int64_t size, double* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str = false) {
    double* dataA = new double[size];
    std::copy(data, data + size, dataA);
    double* A = negative_str ? dataA + size - 1 : dataA;
    for (int i = 0; i < nmode; i++)
    {
        A += offset[i] * strides[i];
    }
    return {A, dataA};
}

std::tuple<std::complex<float>*, std::complex<float>*> copy_tensor_data_c(int64_t size, std::complex<float>* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str = false) {
    std::complex<float>* dataA = new std::complex<float>[size];
    std::copy(data, data + size, dataA);
    std::complex<float>* A = negative_str ? dataA + size - 1 : dataA;
    for (int i = 0; i < nmode; i++)
    {
        A += offset[i] * strides[i];
    }
    return {A, dataA};
}

std::tuple<std::complex<double>*, std::complex<double>*> copy_tensor_data_z(int64_t size, std::complex<double>* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str = false) {
    std::complex<double>* dataA = new std::complex<double>[size];
    std::copy(data, data + size, dataA);
    std::complex<double>* A = negative_str ? dataA + size - 1 : dataA;
    for (int i = 0; i < nmode; i++)
    {
        A += offset[i] * strides[i];
    }
    return {A, dataA};
}

float* copy_tensor_data_s(int size, float* data) {
    float* dataA = new float[size];
    std::copy(data, data + size, dataA);
    return dataA;
}

int calculate_tensor_size(int nmode, int* extents) {
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    return size;
}

std::string str(bool b) {
    return b ? "true" : "false";
}

int randi(int min, int max) {
    return rand() % (max - min + 1) + min;
}

float randf(float min, float max) {
    return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(max-min)));
}

double randd(double min, double max) {
    return min + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(max-min)));
}

int random_choice(int size, int* choices) {
    return choices[randi(0, size - 1)];
}

std::complex<float> randc(std::complex<float> min, std::complex<float> max) {
    return std::complex<float>(min.real() + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(max.real()-min.real()))), min.imag() + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(max.imag()-min.imag()))));
}

std::complex<double> randz(std::complex<double> min, std::complex<double> max) {
    return std::complex<double>(min.real() + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(max.real()-min.real()))), min.imag() + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(max.imag()-min.imag()))));
}

float randf() {
    return (rand() + static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * (rand() % 2 == 0 ? 1 : -1);
}

double randd() {
    return (rand() + static_cast <double> (rand()) / static_cast <double> (RAND_MAX)) * (rand() % 2 == 0 ? 1 : -1);
}

std::complex<float> randc() {
    return std::complex<float>(randf(), randf());
}

std::complex<double> randz() {
    return std::complex<double>(randd(), randd());
}

char* swap_indices(char* indices, int nmode_A, int nmode_B, int nmode_D) {
    char* swapped = new char[nmode_A + nmode_B + nmode_D + 7];
    for (int i = 0; i < nmode_B; i++)
    {
        swapped[i] = indices[nmode_A + 2 + i];
    }
    swapped[nmode_B] = ',';
    swapped[nmode_B+1] = ' ';
    for (int i = 0; i < nmode_A; i++)
    {
        swapped[i + nmode_B + 2] = indices[i];
    }
    swapped[nmode_A+nmode_B+2] = ' ';
    swapped[nmode_A+nmode_B+3] = '-';
    swapped[nmode_A+nmode_B+4] = '>';
    swapped[nmode_A+nmode_B+5] = ' ';
    for (int i = 0; i < nmode_D; i++)
    {
        swapped[i + nmode_B + nmode_A + 6] = indices[nmode_A + nmode_B + 6 + i];
    }
    swapped[nmode_A+nmode_B+nmode_D+6] = '\0';
    return swapped;
}

void rotate_indices(int64_t* idx, int nmode, int64_t* extents, int64_t* strides) {
    if (nmode < 2)
    {
        return;
    }
    int64_t tmp_idx = idx[0];
    int64_t tmp_ext = extents[0];
    int64_t tmp_str = strides[0];
    strides[0] = 1 + ((strides[1] / strides[0]) - extents[0]);
    for (int i = 0; i < nmode - 1; i++)
    {
        idx[i] = idx[i+1];
        if (i == 0)
        {
            strides[i] = 1 * (1 + ((strides[i+1] / strides[i]) - extents[i]));
        }
        else
        {
            strides[i] = strides[i-1] * (extents[i-1] + ((strides[i+1] / strides[i]) - extents[i]));
        }
        extents[i] = extents[i+1];
    }
    idx[nmode-1] = tmp_idx;
    extents[nmode-1] = tmp_ext;
    strides[nmode-1] = strides[nmode-2] * (extents[nmode-2] + (tmp_str - 1));
}

void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents) {
    if (nmode <= 0) {
        return;
    }

    int k = 0;
    do
    {
        coordinates[k] = (coordinates[k] + 1) % extents[k];
        k++;
    } while (coordinates[k - 1] == 0 && k < nmode);
}

void print_tensor_s(int nmode, int64_t* extents, int64_t* strides, float* data) {
    std::cout << "ndim: " << nmode << std::endl;
    std::cout << "extents: ";
    for (int i = 0; i < nmode; i++)
    {
        std::cout << extents[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "strides: ";
    for (int i = 0; i < nmode; i++)
    {
        std::cout << strides[i] << " ";
    }
    std::cout << std::endl;
    int coord[nmode];
    for (int i = 0; i < nmode; i++)
    {
        coord[i] = 0;
    }
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    for (int i = 0; i < size; i++)
    {
        std::cout << data[i] << " ";
        coord[0]++;
        for (int j = 0; j < nmode - 1; j++)
        {
            if (coord[j] == extents[j])
            {
                coord[j] = 0;
                coord[j+1]++;
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

void print_tensor_d(int nmode, int64_t* extents, int64_t* strides, double* data) {
    std::cout << "ndim: " << nmode << std::endl;
    std::cout << "extents: ";
    for (int i = 0; i < nmode; i++)
    {
        std::cout << extents[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "strides: ";
    for (int i = 0; i < nmode; i++)
    {
        std::cout << strides[i] << " ";
    }
    std::cout << std::endl;
    int coord[nmode];
    for (int i = 0; i < nmode; i++)
    {
        coord[i] = 0;
    }
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    for (int i = 0; i < size; i++)
    {
        std::cout << data[i] << " ";
        coord[0]++;
        for (int j = 0; j < nmode - 1; j++)
        {
            if (coord[j] == extents[j])
            {
                coord[j] = 0;
                coord[j+1]++;
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

void print_tensor_c(int nmode, int64_t* extents, int64_t* strides, std::complex<float>* data) {
    std::cout << "ndim: " << nmode << std::endl;
    std::cout << "extents: ";
    for (int i = 0; i < nmode; i++)
    {
        std::cout << extents[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "strides: ";
    for (int i = 0; i < nmode; i++)
    {
        std::cout << strides[i] << " ";
    }
    std::cout << std::endl;
    int coord[nmode];
    for (int i = 0; i < nmode; i++)
    {
        coord[i] = 0;
    }
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    for (int i = 0; i < size; i++)
    {
        std::cout << data[i] << " ";
        coord[0]++;
        for (int j = 0; j < nmode - 1; j++)
        {
            if (coord[j] == extents[j])
            {
                coord[j] = 0;
                coord[j+1]++;
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

void print_tensor_z(int nmode, int64_t* extents, int64_t* strides, std::complex<double>* data) {
    std::cout << "ndim: " << nmode << std::endl;
    std::cout << "extents: ";
    for (int i = 0; i < nmode; i++)
    {
        std::cout << extents[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "strides: ";
    for (int i = 0; i < nmode; i++)
    {
        std::cout << strides[i] << " ";
    }
    std::cout << std::endl;
    int coord[nmode];
    for (int i = 0; i < nmode; i++)
    {
        coord[i] = 0;
    }
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    for (int i = 0; i < size; i++)
    {
        std::cout << data[i] << " ";
        coord[0]++;
        for (int j = 0; j < nmode - 1; j++)
        {
            if (coord[j] == extents[j])
            {
                coord[j] = 0;
                coord[j+1]++;
                std::cout << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

void add_incorrect_idx(int64_t max_idx, int* nmode, int64_t** idx, int64_t** extents, int64_t** strides) {
    int nmode_tmp = *nmode + randi(1, 5);
    int64_t* idx_tmp = new int64_t[nmode_tmp];
    int64_t* extents_tmp = new int64_t[nmode_tmp];
    int64_t* strides_tmp = new int64_t[nmode_tmp];
    std::copy(*idx, *idx + *nmode, idx_tmp);
    std::copy(*extents, *extents + *nmode, extents_tmp);
    std::copy(*strides, *strides + *nmode, strides_tmp);
    for (size_t i = 0; i < nmode_tmp - *nmode; i++)
    {
        idx_tmp[*nmode + i] = max_idx + 1 + i;
    }
    for (size_t i = 0; i < nmode_tmp - *nmode; i++)
    {
        extents_tmp[*nmode + i] = max_idx + 1 + i;
    }
    for (size_t i = 0; i < nmode_tmp - *nmode; i++)
    {
        strides_tmp[*nmode + i] = max_idx + 1 + i;
    }
    delete[] *idx;
    delete[] *extents;
    delete[] *strides;
    *nmode = nmode_tmp;
    *idx = idx_tmp;
    *extents = extents_tmp;
    *strides = strides_tmp;
}

void add_idx(int* nmode, int64_t** idx, int64_t** extents, int64_t** strides, int64_t additional_idx, int64_t additional_extents, int64_t additional_strides) {
    int nmode_tmp = *nmode + 1;
    int64_t* idx_tmp = new int64_t[nmode_tmp];
    int64_t* extents_tmp = new int64_t[nmode_tmp];
    int64_t* strides_tmp = new int64_t[nmode_tmp];
    std::copy(*idx, *idx + *nmode, idx_tmp);
    std::copy(*extents, *extents + *nmode, extents_tmp);
    std::copy(*strides, *strides + *nmode, strides_tmp);
    idx_tmp[*nmode] = additional_idx;
    extents_tmp[*nmode] = additional_extents;
    strides_tmp[*nmode] = additional_strides;
    delete[] *idx;
    delete[] *extents;
    delete[] *strides;
    *nmode = nmode_tmp;
    *idx = idx_tmp;
    *extents = extents_tmp;
    *strides = strides_tmp;
}

bool test_hadamard_product() {
    int nmode = randi(0, 4);
    int64_t* extents = new int64_t[nmode];
    int64_t* strides = new int64_t[nmode];
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        extents[i] = randi(1, 4);
        size *= extents[i];
    }
    if (nmode > 0) {
        strides[0] = 1;
    }
    for (int i = 1; i < nmode; i++)
    {
        strides[i] = strides[i-1] * extents[i-1];
    }
    float* A = new float[size];
    float* B = new float[size];
    float* C = new float[size];
    float* D = new float[size];
    for (int i = 0; i < size; i++)
    {
        A[i] = randf(0, 1);
        B[i] = randf(0, 1);
        C[i] = randf(0, 1);
        D[i] = randf(0, 1);
    }

    float alpha = randf(0, 1);
    float beta = randf(0, 1);

    int64_t* idx_A = new int64_t[nmode];
    for (int i = 0; i < nmode; i++)
    {
        idx_A[i] = 'a' + i;
    }
    int64_t* idx_B = new int64_t[nmode];
    int64_t* idx_C = new int64_t[nmode];
    int64_t* idx_D = new int64_t[nmode];
    std::copy(idx_A, idx_A + nmode, idx_B);
    std::copy(idx_A, idx_A + nmode, idx_C);
    std::copy(idx_A, idx_A + nmode, idx_D);

    float* E = copy_tensor_data_s(size, D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode, extents, strides);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode, extents, strides);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode, extents, strides);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode, extents, strides);

    int op_A = 0;
    int op_B = 0;
    int op_C = 0;
    int op_D = 0;

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode, extents, strides, A, op_A, idx_A,
                   nmode, extents, strides, B, op_B, idx_B,
                   nmode, extents, strides, C, op_C, idx_D,
                   nmode, extents, strides, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(D, E, size);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents;
    delete[] strides;
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;

    return result;
}

bool test_contraction() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s();

    auto [E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_commutativity() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s();

    auto [E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);

    auto [F, data_F] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);

    auto [G, data_G] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);

    auto [C2, data_C2] = copy_tensor_data_s(size_C, data_C, nmode_C, offset_C, strides_C);

    auto [C3, data_C3] = copy_tensor_data_s(size_C, data_C, nmode_C, offset_C, strides_C);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
    create_handle(&handle);
    TAPP_tensor_product planAB;
    TAPP_create_tensor_product(&planAB, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_tensor_product planBA;
    TAPP_create_tensor_product(&planBA, handle, 0, info_B, idx_B, 0, info_A, idx_A, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(planAB, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C2, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    TAPP_execute_product(planBA, exec, &status, (void*)&alpha, (void*)B, (void*)A, (void*)&beta, (void*)C, (void*)F);

    run_tblis_mult_s(nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_C, extents_C, strides_C, C3, 0, idx_D,
                   nmode_D, extents_D, strides_D, G, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D) && compare_tensors_s(data_F, data_G, size_D) && compare_tensors_s(data_D, data_F, size_D);
    
    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(planAB);
    TAPP_destory_tensor_product(planBA);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_C2;
    delete[] data_C3;
    delete[] data_D;
    delete[] data_E;
    delete[] data_F;
    delete[] data_G;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_permutations() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s();
          
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);
    
    bool result = true;

    for (int i = 0; i < nmode_D; i++)
    {
        auto [C2, copy_C2] = copy_tensor_data_s(size_C, data_C, nmode_C, offset_C, strides_C);
        TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
        TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);
        run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                    nmode_B, extents_B, strides_B, B, 0, idx_B,
                    nmode_C, extents_C, strides_C, C2, 0, idx_D,
                    nmode_D, extents_D, strides_D, E, 0, idx_D,
                    alpha, beta);
        result = result && compare_tensors_s(data_D, data_E, size_D);
        rotate_indices(idx_D, nmode_D, extents_D, strides_D);
        rotate_indices(idx_C, nmode_C, extents_C, strides_C);
        TAPP_destory_tensor_product(plan);
        delete[] copy_C2;
    }
    
    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_equal_extents() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_outer_product() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(0, 4), 0);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;
    
    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_full_contraction() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, 0);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_zero_dim_tensor_contraction() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(0);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_one_dim_tensor_contraction() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(1);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_subtensor_same_idx() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_subtensor_lower_idx() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), false, true, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_negative_strides() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), false, false, false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D, true);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return true;
}

bool test_negative_strides_subtensor_same_idx() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), false, true, false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D, true);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_negative_strides_subtensor_lower_idx() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), false, true, true, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D, true);
    
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_contraction_double_precision() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_d();

    auto [E, data_E] = copy_tensor_data_d(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_d(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_d(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_contraction_complex() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_c();

    auto [E, data_E] = copy_tensor_data_c(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C32, nmode_D, extents_D, strides_D);

    int op_A = randi(0, 1);
    int op_B = randi(0, 1);
    int op_C = randi(0, 1);
    int op_D = randi(0, 1);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_c(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors_c(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;
        
    return result;
}

bool test_contraction_complex_double_precision() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_z();

    auto [E, data_E] = copy_tensor_data_z(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C64, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C64, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C64, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C64, nmode_D, extents_D, strides_D);

    int op_A = randi(0, 1);
    int op_B = randi(0, 1);
    int op_C = randi(0, 1);
    int op_D = randi(0, 1);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_z(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                     nmode_B, extents_B, strides_B, B, op_B, idx_B,
                     nmode_C, extents_C, strides_C, C, op_C, idx_D,
                     nmode_D, extents_D, strides_D, E, op_D, idx_D,
                     alpha, beta);

    bool result = compare_tensors_z(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_zero_stride() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s();

    auto [E, data_E] = copy_tensor_data_s(size_D, data_D, nmode_D, offset_D, strides_D);

    if (nmode_A > 0) {
        strides_A[0] = 0;
    }
    else {
        strides_B[0] = 0;
    }

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] data_E;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return result;
}

bool test_error_too_many_idx() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s();

    int64_t max_idx = 0;
    for (size_t i = 0; i < nmode_A; i++)
    {
        if (max_idx < idx_A[i]) {
            max_idx = idx_A[i];
        }
    }
    for (size_t i = 0; i < nmode_B; i++)
    {
        if (max_idx < idx_B[i]) {
            max_idx = idx_B[i];
        }
    }
    for (size_t i = 0; i < nmode_D; i++)
    {
        if (max_idx < idx_D[i]) {
            max_idx = idx_D[i];
        }
    }

    int random_skewed_tensor = randi(0, 2);

    switch (random_skewed_tensor)
    {
    case 0:
        add_incorrect_idx(max_idx, &nmode_A, &idx_A, &extents_A, &strides_A);
        break;
    case 1:
        add_incorrect_idx(max_idx, &nmode_B, &idx_B, &extents_B, &strides_B);
        break;
    case 2:
        add_incorrect_idx(max_idx, &nmode_D, &idx_D, &extents_D, &strides_D);
        break;
    
    default:
        break;
    }

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return (random_skewed_tensor == 0 && error_status == 7) ||
           (random_skewed_tensor == 1 && error_status == 8) ||
           (random_skewed_tensor == 2 && error_status == 9);
}

bool test_error_repeated_idx() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(1, 4));
    
    int nr_choices = 0;
    if (nmode_A > 0) nr_choices++;
    if (nmode_B > 0) nr_choices++;
    if (nmode_D > 0) nr_choices++;

    int* choices = new int[nr_choices];
    int choice_index = 0;

    if (nmode_A > 0) choices[choice_index++] = 0;
    if (nmode_B > 0) choices[choice_index++] = 1;
    if (nmode_D > 0) choices[choice_index++] = 2;

    int random_skewed_tensor = random_choice(nr_choices, choices);
    delete[] choices;
    int random_index = 0;

    switch (random_skewed_tensor)
    {
    case 0:
        if (nmode_A > 1) {
            random_index = randi(0, nmode_A - 1);
            idx_A[random_index] = random_index == 0 ? idx_A[random_index + 1] : idx_A[random_index - 1];
        }
        else {
            add_idx(&nmode_A, &idx_A, &extents_A, &strides_A, idx_A[0], extents_A[0], strides_A[0]);
        }
        break;
    case 1:
        if (nmode_B > 1) {
            random_index = randi(0, nmode_B - 1);
            idx_B[random_index] = random_index == 0 ? idx_B[random_index + 1] : idx_B[random_index - 1];
        }
        else {
            add_idx(&nmode_B, &idx_B, &extents_B, &strides_B, idx_B[0], extents_B[0], strides_B[0]);
        }
        break;
    case 2:
        if (nmode_D > 1) {
            random_index = randi(0, nmode_D - 1);
            idx_D[random_index] = random_index == 0 ? idx_D[random_index + 1] : idx_D[random_index - 1];
        }
        else {
            add_idx(&nmode_D, &idx_D, &extents_D, &strides_D, idx_D[0], extents_D[0], strides_D[0]);
        }
        break;
    default:
        break;
    }

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return (random_skewed_tensor == 0 && error_status == 1) ||
           (random_skewed_tensor == 1 && error_status == 2) ||
           (random_skewed_tensor == 2 && error_status == 3);
}

bool test_error_non_matching_ext() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(1, 4));
    
    int nr_choices = 0;
    if (nmode_A > 0) nr_choices++;
    if (nmode_B > 0) nr_choices++;
    if (nmode_D > 0) nr_choices++;

    int* choices = new int[nr_choices];
    int choice_index = 0;

    if (nmode_A > 0) choices[choice_index++] = 0;
    if (nmode_B > 0) choices[choice_index++] = 1;
    if (nmode_D > 0) choices[choice_index++] = 2;

    int random_skewed_tensor = random_choice(nr_choices, choices);
    delete[] choices;
    int random_index = 0;

    switch (random_skewed_tensor)
    {
    case 0:
        random_index = randi(0, nmode_A - 1);
        extents_A[random_index] += randi(1, 5);
        break;
    case 1:
        random_index = randi(0, nmode_B - 1);
        extents_B[random_index] += randi(1, 5);
        break;
    case 2:
        random_index = randi(0, nmode_D - 1);
        extents_D[random_index] += randi(1, 5);
        break;
    default:
        break;
    }

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return error_status == 4 || error_status == 5 || error_status == 6;
}

bool test_error_C_other_structure() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(1, 4));

    int64_t max_idx = 0;
    for (size_t i = 0; i < nmode_C; i++)
    {
        if (max_idx < idx_C[i]) {
            max_idx = idx_C[i];
        }
    }

    int random_error = randi(0, 2);
    int random_index = 0;

    switch (random_error)
    {
    case 0:
        add_incorrect_idx(max_idx, &nmode_C, &idx_C, &extents_C, &strides_C);
        break;
    case 1:
        if (nmode_C > 1) {
            random_index = randi(0, nmode_C - 1);
            idx_C[random_index] = random_index == 0 ? idx_C[random_index + 1] : idx_C[random_index - 1];
        }
        else {
            add_idx(&nmode_C, &idx_C, &extents_C, &strides_C, idx_C[0], extents_C[0], strides_C[0]);
        }
        break;
    case 2:
        random_index = nmode_C == 1 ? 0 : randi(0, nmode_C - 1);
        extents_C[random_index] += randi(1, 5);
        break;
    default:
        break;
    }

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return error_status == 11 || error_status == 12 || error_status == 13;
}

bool test_error_non_hadamard_shared_idx() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(2, 4));

    int64_t max_idx = 0;
    for (size_t i = 0; i < nmode_A; i++)
    {
        if (max_idx < idx_A[i]) {
            max_idx = idx_A[i];
        }
    }
    for (size_t i = 0; i < nmode_B; i++)
    {
        if (max_idx < idx_B[i]) {
            max_idx = idx_B[i];
        }
    }
    for (size_t i = 0; i < nmode_D; i++)
    {
        if (max_idx < idx_D[i]) {
            max_idx = idx_D[i];
        }
    }

    add_idx(&nmode_A, &idx_A, &extents_A, &strides_A, max_idx + 1, 0, 0);
    add_idx(&nmode_B, &idx_B, &extents_B, &strides_B, max_idx + 1, 0, 0);
    add_idx(&nmode_D, &idx_D, &extents_D, &strides_D, max_idx + 1, 0, 0);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return error_status == 10;
}

bool test_error_aliasing_within_D() {
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction_s(-1, -1, randi(2, 4));

    int scewed_index = randi(1, nmode_D);
    int signs[2] = {-1, 1};
    strides_D[scewed_index] = random_choice(2, signs) * (strides_D[scewed_index - 1] + 1) * randi(1, extents_D[scewed_index - 1] - 1);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destory_tensor_product(plan);
    TAPP_destory_tensor_info(info_A);
    TAPP_destory_tensor_info(info_B);
    TAPP_destory_tensor_info(info_C);
    TAPP_destory_tensor_info(info_D);
    delete[] extents_A;
    delete[] extents_B;
    delete[] extents_C;
    delete[] extents_D;
    delete[] strides_A;
    delete[] strides_B;
    delete[] strides_C;
    delete[] strides_D;
    delete[] idx_A;
    delete[] idx_B;
    delete[] idx_C;
    delete[] idx_D;
    delete[] data_A;
    delete[] data_B;
    delete[] data_C;
    delete[] data_D;
    delete[] offset_A;
    delete[] offset_B;
    delete[] offset_C;
    delete[] offset_D;

    return error_status == 14;
}