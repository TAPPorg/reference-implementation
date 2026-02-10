#include <iostream>
#include <random>
#include <tuple>
#include <string>
#include <complex>
#include <algorithm>
#include <unordered_map>
#include <type_traits>
#include <dlfcn.h>  // POSIX dynamic loading, TODO: fix for windows

extern "C" {
    #include <tapp.h>
}

const char* pathA = "./libtapp-reference.so";
const char* pathB = "./cutensor_bindings/libcutensor_bindings.so";
struct imp
{
    void* handle;
    TAPP_error (*TAPP_attr_set)(TAPP_attr attr, TAPP_key key, void* value);
    TAPP_error (*TAPP_attr_get)(TAPP_attr attr, TAPP_key key, void* value);
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

void load_implementation(struct imp* imp, const char* path);
void unload_implementation(struct imp* imp);

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
std::tuple<int, int, int, int,
           int, int, int, int,
           int, int, int, int> generate_index_configuration(int nmode_A = -1, int nmode_B = -1, int nmode_D = -1,
                                                            int contracted_indices = -1, int hadamard_indices = -1,
                                                            bool hadamard_only = false, bool hadamard_indices_enabled = false,
                                                            bool isolated_indices_enabled = false, bool repeated_indices_enabled = false);
int* generate_unique_indices(int64_t total_unique_indices);
std::tuple<int64_t*, int64_t*, int64_t*, int64_t*> assign_indices(int* unique_indices,
                                                                  int contracted_modes, int hadamard_modes,
                                                                  int free_indices_A, int free_indices_B,
                                                                  int isolated_indices_A, int isolated_indices_B,
                                                                  int repeated_indices_A, int repeated_indices_B);
std::unordered_map<int, int64_t> generate_index_extent_map(int64_t min_extent, int64_t max_extent,
                                                           bool equal_extents_only,
                                                           int64_t total_unique_indices, int* unique_indices);
std::tuple<int64_t*, int64_t*, int64_t*, int64_t*> assign_extents(std::unordered_map<int, int64_t> index_extent_map,
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
