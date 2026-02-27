#include <iostream>
#include <random>
#include <tuple>
#include <string>
#include <complex>
#include <algorithm>
#include <unordered_map>
#include <type_traits>
#include <dlfcn.h>  // POSIX dynamic loading, TODO: fix for windows

#ifndef TAPP_DYNAMIC_LAUNCH
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tblis.h"
#pragma GCC diagnostic pop
#endif
#include <tapp.h>

#ifdef TAPP_DYNAMIC_LAUNCH
const char* pathA = "./reference_implementation/libtapp-reference.so";
const char* pathB = "./cutensor_bindings/libtapp-cutensor.so";
struct impl
{
    void* handle;
    TAPP_error (*TAPP_attr_set)(TAPP_attr attr, TAPP_key key, void* value);
    TAPP_error (*TAPP_attr_get)(TAPP_attr attr, TAPP_key key, void** value);
    TAPP_error (*TAPP_attr_clear)(TAPP_attr attr, TAPP_key key);
    bool (*TAPP_check_success)(TAPP_error error);
    size_t (*TAPP_explain_error)(TAPP_error error, size_t maxlen, char* message);
    TAPP_error (*TAPP_create_executor)(TAPP_executor* exec);
    TAPP_error (*TAPP_destroy_executor)(TAPP_executor exec);
    TAPP_error (*TAPP_create_handle)(TAPP_handle* handle);
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
                                          TAPP_handle handle,
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

int load_implementation(struct impl* impl, const char* path);
void unload_implementation(struct impl* impl);
#else
template<typename T>
T* run_tblis_mult(int nmode_A, int64_t* extents_A, int64_t* strides_A, T* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, T* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, T* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, T* D, int op_D, int64_t* idx_D,
                    T alpha, T beta);
template<typename T>
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, T*> reduce_isolated_indices(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_X, tblis::label_type* idx_X, int nmode_Y, tblis::label_type* idx_Y);
#endif

template<typename T>
TAPP_error run_product(
#ifdef TAPP_DYNAMIC_LAUNCH
                  struct impl impl, bool use_device_memory,
#else
                  bool use_tblis,
#endif
                  int nmode_A, int64_t* extents_A, int64_t* strides_A, T* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, T* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, T* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, T* D, int op_D, int64_t* idx_D,
                  T alpha, T beta
                  );

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
bool test_hadamard_product(
#ifdef TAPP_DYNAMIC_LAUNCH
                           struct impl implA, struct impl implB
#endif
                           );
bool test_contraction(
#ifdef TAPP_DYNAMIC_LAUNCH
                      struct impl implA, struct impl implB
#endif
                      );
bool test_commutativity(
#ifdef TAPP_DYNAMIC_LAUNCH
                        struct impl implA, struct impl implB
#endif
                        );
bool test_permutations(
#ifdef TAPP_DYNAMIC_LAUNCH
                       struct impl implA, struct impl implB
#endif
                       );
bool test_equal_extents(
#ifdef TAPP_DYNAMIC_LAUNCH
                        struct impl implA, struct impl implB
#endif
                        );
bool test_outer_product(
#ifdef TAPP_DYNAMIC_LAUNCH
                        struct impl implA, struct impl implB
#endif
                        );
bool test_full_contraction(
#ifdef TAPP_DYNAMIC_LAUNCH
                           struct impl implA, struct impl implB
#endif
                           );
bool test_zero_dim_tensor_contraction(
#ifdef TAPP_DYNAMIC_LAUNCH
                                      struct impl implA, struct impl implB
#endif
                                      );
bool test_one_dim_tensor_contraction(
#ifdef TAPP_DYNAMIC_LAUNCH
                                     struct impl implA, struct impl implB
#endif
                                     );
bool test_subtensor_same_idx(
#ifdef TAPP_DYNAMIC_LAUNCH
                             struct impl implA, struct impl implB
#endif
                             );
bool test_subtensor_lower_idx(
#ifdef TAPP_DYNAMIC_LAUNCH
                              struct impl implA, struct impl implB
#endif
                              );
bool test_negative_strides(
#ifdef TAPP_DYNAMIC_LAUNCH
                           struct impl implA, struct impl implB
#endif
                           );
bool test_negative_strides_subtensor_same_idx(
#ifdef TAPP_DYNAMIC_LAUNCH
                                              struct impl implA, struct impl implB
#endif
                                              );
bool test_negative_strides_subtensor_lower_idx(
#ifdef TAPP_DYNAMIC_LAUNCH
                                               struct impl implA, struct impl implB
#endif
                                               );
bool test_mixed_strides(
#ifdef TAPP_DYNAMIC_LAUNCH
                        struct impl implA, struct impl implB
#endif
                        );
bool test_mixed_strides_subtensor_same_idx(
#ifdef TAPP_DYNAMIC_LAUNCH
                                           struct impl implA, struct impl implB
#endif
                                           );
bool test_mixed_strides_subtensor_lower_idx(
#ifdef TAPP_DYNAMIC_LAUNCH
                                            struct impl implA, struct impl implB
#endif
                                            );
bool test_contraction_double_precision(
#ifdef TAPP_DYNAMIC_LAUNCH
                                       struct impl implA, struct impl implB
#endif
                                       );
bool test_contraction_complex(
#ifdef TAPP_DYNAMIC_LAUNCH
                              struct impl implA, struct impl implB
#endif
                              );
bool test_contraction_complex_double_precision(
#ifdef TAPP_DYNAMIC_LAUNCH
                                               struct impl implA, struct impl implB
#endif
                                               );
bool test_zero_stride(
#ifdef TAPP_DYNAMIC_LAUNCH
                      struct impl implA, struct impl implB
#endif
                      );
bool test_unique_idx(
#ifdef TAPP_DYNAMIC_LAUNCH
                     struct impl implA, struct impl implB
#endif
                     );
bool test_repeated_idx(
#ifdef TAPP_DYNAMIC_LAUNCH
                       struct impl implA, struct impl implB
#endif
                       );
bool test_hadamard_and_free(
#ifdef TAPP_DYNAMIC_LAUNCH
                            struct impl implA, struct impl implB
#endif
                            );
bool test_hadamard_and_contraction(
#ifdef TAPP_DYNAMIC_LAUNCH
                                   struct impl implA, struct impl implB
#endif
                                   );

#ifndef TAPP_DYNAMIC_LAUNCH // These test does not make sense for other implementations than the reference
bool test_error_non_matching_ext();
bool test_error_C_other_structure();
bool test_error_aliasing_within_D();
#endif
