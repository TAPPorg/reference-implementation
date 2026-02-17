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
#include <algorithm>
#include <unordered_map>
#include <type_traits>
#include <cstring>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tblis.h"
#pragma GCC diagnostic pop
#include <tapp.h>

template<typename T>
void run_tblis_mult(int nmode_A, int64_t* extents_A, int64_t* strides_A, T* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, T* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, T* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, T* D, int op_D, int64_t* idx_D,
                    T alpha, T beta);
template<typename T>
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, T*> reduce_isolated_indices(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_X, tblis::label_type* idx_X, int nmode_Y, tblis::label_type* idx_Y);

template<typename T>
struct is_complex : std::false_type {};
template<typename T>
struct is_complex<std::complex<T>> : std::true_type {};
template<typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

template<typename T>
T rand(T min, T max);
template<typename T>
T rand();

template<typename T, typename U>
U* change_array_type(T* array, int size);
template<typename T>
bool compare_tensors(T* A, T* B, int64_t size);
template<typename T>
std::tuple<int, int64_t*, int64_t*, T*, int64_t*,
           int, int64_t*, int64_t*, T*, int64_t*,
           int, int64_t*, int64_t*, T*, int64_t*,
           int, int64_t*, int64_t*, T*, int64_t*,
           T, T,
           T*, T*, T*, T*,
           int64_t, int64_t, int64_t, int64_t> generate_pseudorandom_contraction(int nmode_A = -1, int nmode_B = -1,
                                                                                 int nmode_D = -1, int contracted_indices = -1,
                                                                                 int hadamard_indices = -1,
                                                                                 int min_extent = 1, bool equal_extents_only = false,
                                                                                 bool subtensor_on_extents = false, bool subtensor_on_nmode = false,
                                                                                 bool negative_strides_enabled = false, bool mixed_strides_enabled = false,
                                                                                 bool hadamard_indices_enabled = false, bool hadamard_only = false,
                                                                                 bool repeated_indices_enabled = false, bool isolated_indices_enabled = false);
std::tuple<int, int, int,
           int, int, int, int,
           int, int, int, int> generate_index_configuration(int nmode_A = -1, int nmode_B = -1, int nmode_D = -1,
                                                            int contracted_indices = -1, int hadamard_indices = -1,
                                                            bool hadamard_only = false, bool hadamard_indices_enabled = false,
                                                            bool isolated_indices_enabled = false, bool repeated_indices_enabled = false);
int* generate_unique_indices(int64_t total_unique_indices);
std::tuple<int64_t*, int64_t*, int64_t*> assign_indices(int* unique_indices,
                                                        int contracted_modes, int hadamard_modes,
                                                        int free_indices_A, int free_indices_B,
                                                        int isolated_indices_A, int isolated_indices_B,
                                                        int repeated_indices_A, int repeated_indices_B);
std::unordered_map<int, int64_t> generate_index_extent_map(int64_t min_extent, int64_t max_extent,
                                                           bool equal_extents_only,
                                                           int64_t total_unique_indices, int* unique_indices);
std::tuple<int64_t*, int64_t*, int64_t*> assign_extents(std::unordered_map<int, int64_t> index_extent_map,
                                                        int nmode_A, int64_t* idx_A,
                                                        int nmode_B, int64_t* idx_B,
                                                        int nmode_D, int64_t* idx_D);
int* choose_stride_signs(int nmode, bool negative_str, bool mixed_str);
bool* choose_subtensor_dims(int nmode, int outer_nmode);
int64_t* calculate_outer_extents(int outer_nmode, int64_t* extents, bool* subtensor_dims, bool lower_extents);
int64_t* calculate_offsets(int nmode, int outer_nmode, int64_t* extents, int64_t* outer_extents, bool* subtensor_dims, bool lower_extents);
int64_t* calculate_strides(int nmode, int outer_nmode, int64_t* outer_extents, int* stride_signs, bool* subtensor_dims);
int calculate_size(int nmode, int64_t* extents);
template<typename T>
T* create_tensor_data(int64_t size);
template<typename T>
T* create_tensor_data(int64_t size, T min_value, T max_value);
template<typename T>
T* calculate_tensor_pointer(T* pointer, int nmode, int64_t* extents, int64_t* offsets, int64_t* strides);
void* calculate_tensor_pointer(void* pointer, int nmode, int64_t* extents, int64_t* offsets, int64_t* strides, unsigned long data_size);
template<typename T>
std::tuple<T*, T*> copy_tensor_data(int64_t size, T* data, T* pointer);
template<typename T>
T* copy_tensor_data(int64_t size, T* data);
int calculate_tensor_size(int nmode, int* extents);
template<typename T>
T random_choice(int size, T* choices);
char* swap_indices(char* indices, int nmode_A, int nmode_B, int nmode_D);
void rotate_indices(int64_t* idx, int nmode, int64_t* extents, int64_t* strides);
void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents);
void print_tensor(int nmode, int64_t* extents, int64_t* strides);
template<typename T>
void print_tensor(int nmode, int64_t* extents, int64_t* strides, T* data);
void add_incorrect_idx(int64_t max_idx, int* nmode, int64_t** idx, int64_t** extents, int64_t** strides);
void add_idx(int* nmode, int64_t** idx, int64_t** extents, int64_t** strides, int64_t additional_idx, int64_t additional_extents, int64_t additional_strides);

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
bool test_subtensor_unchanged_nmode();
bool test_subtensor_lower_nmode();
bool test_negative_strides();
bool test_negative_strides_subtensor_unchanged_nmode();
bool test_negative_strides_subtensor_lower_nmode();
bool test_mixed_strides();
bool test_mixed_strides_subtensor_unchanged_nmode();
bool test_mixed_strides_subtensor_lower_nmode();
bool test_contraction_double_precision();
bool test_contraction_complex();
bool test_contraction_complex_double_precision();
bool test_zero_stride();
bool test_isolated_idx();
bool test_repeated_idx();
bool test_hadamard_and_free();
bool test_hadamard_and_contraction();
bool test_error_non_matching_ext();
bool test_error_C_other_structure();
bool test_error_aliasing_within_D();
