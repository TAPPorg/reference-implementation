#include <iostream>
#include <random>
#include <tuple>
#include <string>
#include <complex>
#include <algorithm>
#include <dlfcn.h>  // POSIX dynamic loading, TODO: fix for windows
extern "C" {
    #include "tapp_ex_imp.h"
}

const char* pathA = "libtapp.so";
const char* pathB = "libcutensor_binds.so";
struct imp
{
    void* handle;
    TAPP_error (*TAPP_attr_set)(TAPP_attr attr, TAPP_key key, void* value);
    TAPP_error (*TAPP_attr_get)(TAPP_attr attr, TAPP_key key, void** value);
    TAPP_error (*TAPP_attr_clear)(TAPP_attr attr, TAPP_key key);
    bool (*TAPP_check_success)(TAPP_error error);
    size_t (*TAPP_explain_error)(TAPP_error error, size_t maxlen, char* message);
    TAPP_error (*create_executor)(TAPP_executor* exec);
    TAPP_error (*TAPP_destroy_executor)(TAPP_executor exec);
    TAPP_error (*create_handle)(TAPP_handle* handle);
    TAPP_error (*TAPP_destroy_handle)(TAPP_handle handle);
    TAPP_error (*TAPP_create_tensor_product)(TAPP_tensor_product* plan,
                                             TAPP_handle handle,
                                             TAPP_element_op op_A,
                                             TAPP_tensor_info A,
                                             const int64_t* idx_A,
                                             TAPP_element_op op_B,
                                             TAPP_tensor_info B,
                                             const int64_t* idx_B,
                                             TAPP_element_op op_C,
                                             TAPP_tensor_info C,
                                             const int64_t* idx_C,
                                             TAPP_element_op op_D,
                                             TAPP_tensor_info D,
                                             const int64_t* idx_D,
                                             TAPP_prectype prec);
    TAPP_error (*TAPP_destroy_tensor_product)(TAPP_tensor_product plan);
    TAPP_error (*TAPP_execute_product)(TAPP_tensor_product plan,
                                       TAPP_executor exec,
                                       TAPP_status* status,
                                       const void* alpha,
                                       const void* A,
                                       const void* B,
                                       const void* beta,
                                       const void* C,
                                             void* D);
    TAPP_error (*TAPP_execute_batched_product)(TAPP_tensor_product plan,
                                               TAPP_executor exec,
                                               TAPP_status* status,
                                               int num_batches,
                                               const void* alpha,
                                               const void** A,
                                               const void** B,
                                               const void* beta,
                                               const void** C,
                                                     void** D);
    TAPP_error (*TAPP_destroy_status)(TAPP_status status);
    TAPP_error (*TAPP_create_tensor_info)(TAPP_tensor_info* info,
                                          TAPP_datatype type,
                                          int nmode,
                                          const int64_t* extents,
                                          const int64_t* strides);
    TAPP_error (*TAPP_destroy_tensor_info)(TAPP_tensor_info info);
    int (*TAPP_get_nmodes)(TAPP_tensor_info info);
    TAPP_error (*TAPP_set_nmodes)(TAPP_tensor_info info, int nmodes);
    void (*TAPP_get_extents)(TAPP_tensor_info info, int64_t* extents);
    TAPP_error (*TAPP_set_extents)(TAPP_tensor_info info, const int64_t* extents);
    void (*TAPP_get_strides)(TAPP_tensor_info info, int64_t* strides);
    TAPP_error (*TAPP_set_strides)(TAPP_tensor_info info, const int64_t* strides);
};

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
float* create_tensor_data_s(int64_t size);
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
std::complex<float>* create_tensor_data_c(int64_t size);

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
std::complex<double>* create_tensor_data_z(int64_t size);



std::string str(bool b);
int randi(int min, int max);
char* swap_indices(char* indices, int nmode_A, int nmode_B, int nmode_D);
void add_incorrect_idx(int64_t max_idx, int* nmode, int64_t** idx, int64_t** extents, int64_t** strides);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
int* choose_stride_signs(int nmode, bool negative_str, bool mixed_str);
bool* choose_subtensor_dims(int nmode, int outer_nmode);
int64_t* calculate_outer_extents(int outer_nmode, int64_t* extents, bool* subtensor_dims, bool lower_extents);
int64_t* calculate_offsets(int nmode, int outer_nmode, int64_t* extents, int64_t* outer_extents, bool* subtensor_dims, bool lower_extents);
int64_t* calculate_strides(int nmode, int outer_nmode, int64_t* outer_extents, int* stride_signs, bool* subtensor_dims);
int calculate_size(int nmode, int64_t* extents);
void* calculate_tensor_pointer(void* pointer, int nmode, int64_t* extents, int64_t* offsets, int64_t* strides, unsigned long data_size);

void load_implementation(struct imp* imp, const char* path);
void unload_implementation(struct imp* imp);

// Tests
bool test_hadamard_product(struct imp impA, struct imp impB);
bool test_contraction(struct imp impA, struct imp impB);
bool test_commutativity(struct imp impA, struct imp impB);
bool test_permutations(struct imp impA, struct imp impB);
bool test_equal_extents(struct imp impA, struct imp impB);
bool test_outer_product(struct imp impA, struct imp impB);
bool test_full_contraction(struct imp impA, struct imp impB);
bool test_zero_dim_tensor_contraction(struct imp impA, struct imp impB);
bool test_one_dim_tensor_contraction(struct imp impA, struct imp impB);
bool test_subtensor_same_idx(struct imp impA, struct imp impB);
bool test_subtensor_lower_idx(struct imp impA, struct imp impB);
bool test_negative_strides(struct imp impA, struct imp impB);
bool test_negative_strides_subtensor_same_idx(struct imp impA, struct imp impB);
bool test_negative_strides_subtensor_lower_idx(struct imp impA, struct imp impB);
bool test_mixed_strides(struct imp impA, struct imp impB);
bool test_mixed_strides_subtensor_same_idx(struct imp impA, struct imp impB);
bool test_mixed_strides_subtensor_lower_idx(struct imp impA, struct imp impB);
bool test_contraction_double_precision(struct imp impA, struct imp impB);
bool test_contraction_complex(struct imp impA, struct imp impB);
bool test_contraction_complex_double_precision(struct imp impA, struct imp impB);
bool test_zero_stride(struct imp impA, struct imp impB);
bool test_unique_idx(struct imp impA, struct imp impB);
bool test_repeated_idx(struct imp impA, struct imp impB);
bool test_hadamard_and_free(struct imp impA, struct imp impB);
bool test_hadamard_and_contraction(struct imp impA, struct imp impB);
bool test_error_non_matching_ext(struct imp impA, struct imp impB);
bool test_error_C_other_structure(struct imp impA, struct imp impB);
bool test_error_aliasing_within_D(struct imp impA, struct imp impB);
