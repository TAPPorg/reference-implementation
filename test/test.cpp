/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#include "test.h"

// TODO replace by #include of <blis.h> when possible
extern "C" {
  extern void bli_init();
  extern void bli_finalize();
}

unsigned int current_rand_seed = 0;
auto& rand_engine() {
    static std::mt19937 engine(current_rand_seed);
    return engine;
}

int main(int argc, char const *argv[])
{
    if (argc >= 2) current_rand_seed = std::atoi(argv[1]); // now ready to generate random numbers
    bli_init();
    std::cout << std::boolalpha;
    std::cout << "Starting seed for random numbers = " << current_rand_seed << std::endl;
    std::cout << "Hadamard Product: " << test_hadamard_product() << std::endl;
    std::cout << "Contraction: " << test_contraction() << std::endl;
    std::cout << "Commutativity: " << test_commutativity() << std::endl;
    std::cout << "Permutations: " << test_permutations() << std::endl;
    std::cout << "Equal Extents: " << test_equal_extents() << std::endl;
    std::cout << "Outer Product: " << test_outer_product() << std::endl;
    std::cout << "Full Contraction: " << test_full_contraction() << std::endl;
    //for(int i=0;i<0;i++)
    std::cout << "Zero Dim Tensor Contraction: " << test_zero_dim_tensor_contraction() << std::endl;
    std::cout << "One Dim Tensor Contraction: " << test_one_dim_tensor_contraction() << std::endl;
    std::cout << "Subtensor Same Nmode: " << test_subtensor_unchanged_nmode() << std::endl;
    std::cout << "Subtensor Lower Nmode: " << test_subtensor_lower_nmode() << std::endl;
    std::cout << "Negative Strides: " << test_negative_strides() << std::endl;
    std::cout << "Negative Strides Subtensor Same Nmode: " << test_negative_strides_subtensor_unchanged_nmode() << std::endl;
    std::cout << "Negative Strides Subtensor Lower Nmode: " << test_negative_strides_subtensor_lower_nmode() << std::endl;
    std::cout << "Mixed Strides: " << test_mixed_strides() << std::endl;
    std::cout << "Mixed Strides Subtensor Same Nmode: " << test_mixed_strides_subtensor_unchanged_nmode() << std::endl;
    std::cout << "Mixed Strides Subtensor Lower Nmode: " << test_mixed_strides_subtensor_lower_nmode() << std::endl;
    std::cout << "Contraction Double Precision: " << test_contraction_double_precision() << std::endl;
    std::cout << "Contraction Complex: " << test_contraction_complex() << std::endl;
    //for(int i=0;i<1;i++)
    std::cout << "Contraction Complex Double Precision: " << test_contraction_complex_double_precision() << std::endl;
    std::cout << "Zero stride: " << test_zero_stride() << std::endl;
    std::cout << "Isolated Indices: " << test_isolated_idx() << std::endl;
    std::cout << "Repeated Indices: " << test_repeated_idx() << std::endl;
    std::cout << "Hadamard And Free: " << test_hadamard_and_free() << std::endl;
    std::cout << "Hadamard And Contraction: " << test_hadamard_and_contraction() << std::endl;
    std::cout << "Error: Non Matching Extents: " << test_error_non_matching_ext() << std::endl;
    std::cout << "Error: C Other Structure: " << test_error_C_other_structure() << std::endl;
    std::cout << "Error: Aliasing Within D: " << test_error_aliasing_within_D() << std::endl;
    bli_finalize();
    return 0;
}

template<typename T>
void run_tblis_mult(int nmode_A, int64_t* extents_A, int64_t* strides_A, T* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, T* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, T* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, T* D, int op_D, int64_t* idx_D,
                    T alpha, T beta)
{
    tblis::len_type* tblis_len_A = change_array_type<int64_t, tblis::len_type>(extents_A, nmode_A);
    tblis::stride_type* tblis_stride_A = change_array_type<int64_t, tblis::stride_type>(strides_A, nmode_A);
    tblis::tblis_tensor tblis_A;
    tblis::label_type* tblis_idx_A = change_array_type<int64_t, tblis::label_type>(idx_A, nmode_A);

    tblis::len_type* tblis_len_B = change_array_type<int64_t, tblis::len_type>(extents_B, nmode_B);
    tblis::stride_type* tblis_stride_B = change_array_type<int64_t, tblis::stride_type>(strides_B, nmode_B);
    tblis::tblis_tensor tblis_B;
    tblis::label_type* tblis_idx_B = change_array_type<int64_t, tblis::label_type>(idx_B, nmode_B);

    tblis::len_type* tblis_len_C = change_array_type<int64_t, tblis::len_type>(extents_C, nmode_C);
    tblis::stride_type* tblis_stride_C = change_array_type<int64_t, tblis::stride_type>(strides_C, nmode_C);
    tblis::tblis_tensor tblis_C;
    tblis::label_type* tblis_idx_C = change_array_type<int64_t, tblis::label_type>(idx_C, nmode_C);
    
    tblis::len_type* tblis_len_D = change_array_type<int64_t, tblis::len_type>(extents_D, nmode_D);
    tblis::stride_type* tblis_stride_D = change_array_type<int64_t, tblis::stride_type>(strides_D, nmode_D);
    tblis::tblis_tensor tblis_D;
    tblis::label_type* tblis_idx_D = change_array_type<int64_t, tblis::label_type>(idx_D, nmode_D);

    if constexpr (std::is_same_v<T, float>)
    {
        tblis_init_tensor_scaled_s(&tblis_A, alpha, nmode_A, tblis_len_A, A, tblis_stride_A);
        tblis_init_tensor_s(&tblis_B, nmode_B, tblis_len_B, B, tblis_stride_B);
        tblis_init_tensor_scaled_s(&tblis_C, beta, nmode_C, tblis_len_C, C, tblis_stride_C);
        tblis_init_tensor_scaled_s(&tblis_D, 0, nmode_D, tblis_len_D, D, tblis_stride_D);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        tblis_init_tensor_scaled_d(&tblis_A, alpha, nmode_A, tblis_len_A, A, tblis_stride_A);
        tblis_init_tensor_d(&tblis_B, nmode_B, tblis_len_B, B, tblis_stride_B);
        tblis_init_tensor_scaled_d(&tblis_C, beta, nmode_C, tblis_len_C, C, tblis_stride_C);
        tblis_init_tensor_scaled_d(&tblis_D, 0, nmode_D, tblis_len_D, D, tblis_stride_D);
    }
    else if constexpr (is_complex_v<T>) 
    {
        using value_type = typename T::value_type;
        if constexpr (std::is_same_v<value_type, float>)
        {
            tblis_init_tensor_scaled_c(&tblis_A, alpha, nmode_A, tblis_len_A, A, tblis_stride_A);
            tblis_init_tensor_c(&tblis_B, nmode_B, tblis_len_B, B, tblis_stride_B);
            tblis_init_tensor_scaled_c(&tblis_C, beta, nmode_C, tblis_len_C, C, tblis_stride_C);
            tblis_init_tensor_scaled_c(&tblis_D, 0, nmode_D, tblis_len_D, D, tblis_stride_D);
        }
        else if constexpr (std::is_same_v<value_type, double>)
        {
            tblis_init_tensor_scaled_z(&tblis_A, alpha, nmode_A, tblis_len_A, A, tblis_stride_A);
            tblis_init_tensor_z(&tblis_B, nmode_B, tblis_len_B, B, tblis_stride_B);
            tblis_init_tensor_scaled_z(&tblis_C, beta, nmode_C, tblis_len_C, C, tblis_stride_C);
            tblis_init_tensor_scaled_z(&tblis_D, 0, nmode_D, tblis_len_D, D, tblis_stride_D);
        }
    }

    auto [tblis_A_reduced, tblis_idx_A_reduced, tblis_len_A_reduced, tblis_stride_A_reduced, tblis_data_A_reduced] = contract_unique_idx<T>(&tblis_A, tblis_idx_A, nmode_B, tblis_idx_B, nmode_D, tblis_idx_D);

    auto [tblis_B_reduced, tblis_idx_B_reduced, tblis_len_B_reduced, tblis_stride_B_reduced, tblis_data_B_reduced] = contract_unique_idx<T>(&tblis_B, tblis_idx_B, nmode_A, tblis_idx_A, nmode_D, tblis_idx_D);    

    tblis_tensor_mult(tblis_single, NULL, tblis_A_reduced, tblis_idx_A_reduced, tblis_B_reduced, tblis_idx_B_reduced, &tblis_D, tblis_idx_D);

    tblis_tensor_add(tblis_single, NULL, &tblis_C, tblis_idx_C, &tblis_D, tblis_idx_D);

    delete[] tblis_idx_A;
    delete[] tblis_len_A;
    delete[] tblis_stride_A;

    delete[] tblis_idx_B;
    delete[] tblis_len_B;
    delete[] tblis_stride_B;

    delete[] tblis_idx_C;
    delete[] tblis_len_C;
    delete[] tblis_stride_C;

    delete[] tblis_idx_D;
    delete[] tblis_len_D;
    delete[] tblis_stride_D;

    delete[] tblis_idx_A_reduced;
    delete[] tblis_len_A_reduced;
    delete[] tblis_stride_A_reduced;
    delete[] tblis_data_A_reduced;
    delete tblis_A_reduced;

    delete[] tblis_idx_B_reduced;
    delete[] tblis_len_B_reduced;
    delete[] tblis_stride_B_reduced;
    delete[] tblis_data_B_reduced;
    delete tblis_B_reduced;
}

template<typename T>
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, T*> contract_unique_idx(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2)
{
    int nmode_reduced = 0;
    int64_t size_reduced = 1;
    tblis::tblis_tensor* tblis_reduced = new tblis::tblis_tensor;
    tblis::len_type* len_reduced = new tblis::len_type[tensor->ndim];
    tblis::stride_type* stride_reduced = new tblis::stride_type[tensor->ndim];
    tblis::label_type* idx_reduced = new tblis::label_type[tensor->ndim+1];
    for (size_t i = 0; i < tensor->ndim; i++)
    {
        bool found = false;
        for (size_t j = 0; j < nmode_1; j++)
        {
            if (idx[i] == idx_1[j]) 
            {
                found = true;
            }
        }
        for (size_t j = 0; j < nmode_2; j++)
        {
            if (idx[i] == idx_2[j]) 
            {
                found = true;
            }
        }
        
        if (found)
        {
            len_reduced[nmode_reduced] = tensor->len[i];
            stride_reduced[nmode_reduced] = nmode_reduced == 0 ? 1 : stride_reduced[nmode_reduced - 1] * len_reduced[nmode_reduced - 1];
            idx_reduced[nmode_reduced] = idx[i];
            size_reduced *= len_reduced[nmode_reduced];
            nmode_reduced++;
        }
    }
    idx_reduced[nmode_reduced] = '\0';

    T* data_reduced = new T[size_reduced];
    for (size_t i = 0; i < size_reduced; i++)
    {
        data_reduced[i] = 0;
    }

    if constexpr (std::is_same_v<T, float>)
    {
        tblis_init_tensor_s(tblis_reduced, nmode_reduced, len_reduced, data_reduced, stride_reduced);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        tblis_init_tensor_d(tblis_reduced, nmode_reduced, len_reduced, data_reduced, stride_reduced);
    }
    else if constexpr (is_complex_v<T>) 
    {
        using value_type = typename T::value_type;
        if constexpr (std::is_same_v<value_type, float>)
        {
            tblis_init_tensor_c(tblis_reduced, nmode_reduced, len_reduced, data_reduced, stride_reduced);
        }
        else if constexpr (std::is_same_v<value_type, double>)
        {
            tblis_init_tensor_z(tblis_reduced, nmode_reduced, len_reduced, data_reduced, stride_reduced);
        }
    }
    tblis_tensor_add(tblis_single, NULL, tensor, idx, tblis_reduced, idx_reduced);
    return {tblis_reduced, idx_reduced, len_reduced, stride_reduced, data_reduced};
}

template<typename T, typename U>
U* change_array_type(T* array, int size)
{
    U* new_array = new U[size];
    for (int i = 0; i < size; i++)
    {
        new_array[i] = array[i];
    }
    return new_array;
}

template<typename T>
bool compare_tensors(T* A, T* B, int64_t size)
{
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        if constexpr (is_complex_v<T>) 
        {
            using value_type = typename T::value_type;
            value_type rel_diff_r = abs((A[i].real() - B[i].real()) / (A[i].real() > B[i].real() ? A[i].real() : B[i].real()));
            value_type rel_diff_i = abs((A[i].imag() - B[i].imag()) / (A[i].imag() > B[i].imag() ? A[i].imag() : B[i].imag()));
            if (rel_diff_r > 0.00005 || rel_diff_i > 0.00005)
            {
                std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
                std::cout << "\n" << i << ": " << std::complex<value_type>(rel_diff_r, rel_diff_i) << std::endl;
                found = true;
            }
        }
        else
        {
            T rel_diff = abs((A[i] - B[i]) / (A[i] > B[i] ? A[i] : B[i]));
            if (rel_diff > 0.00005)
            {
                std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
                std::cout << "\n" << i << ": " << rel_diff << std::endl;
                found = true;
            }
        }
    }
    return !found;
}

template<typename T>
std::tuple<int, int64_t*, int64_t*, T*, int64_t*,
           int, int64_t*, int64_t*, T*, int64_t*,
           int, int64_t*, int64_t*, T*, int64_t*,
           int, int64_t*, int64_t*, T*, int64_t*,
           T, T,
           T*, T*, T*, T*,
           int64_t, int64_t, int64_t, int64_t> generate_pseudorandom_contraction(int nmode_A, int nmode_B,
                                                                                 int nmode_D, int contracted_indices,
                                                                                 int hadamard_indices,
                                                                                 int min_extent, bool equal_extents_only,
                                                                                 bool subtensor_on_extents, bool subtensor_on_nmode,
                                                                                 bool negative_strides_enabled, bool mixed_strides_enabled,
                                                                                 bool hadamard_indices_enabled, bool hadamard_only,
                                                                                 bool repeated_indices_enabled, bool isolated_indices_enabled)
{
    int nmode_C, free_indices_A, free_indices_B, isolated_indices_A, isolated_indices_B, repeated_indices_A, repeated_indices_B;

    std::tie(nmode_A, nmode_B, nmode_C, nmode_D,
             contracted_indices, hadamard_indices,
             free_indices_A, free_indices_B,
             isolated_indices_A, isolated_indices_B,
             repeated_indices_A, repeated_indices_B) = generate_index_configuration(nmode_A, nmode_B, nmode_D,
                                                                                    contracted_indices, hadamard_indices,
                                                                                    hadamard_only, hadamard_indices_enabled,
                                                                                    isolated_indices_enabled, repeated_indices_enabled);

    int64_t total_unique_indices = contracted_indices + hadamard_indices +
                                   free_indices_A + free_indices_B +
                                   isolated_indices_A + isolated_indices_B +
                                   repeated_indices_A + repeated_indices_B;

    int* unique_indices = generate_unique_indices(total_unique_indices);

    auto [idx_A, idx_B, idx_C, idx_D] = assign_indices(unique_indices,
                                                       contracted_indices, hadamard_indices,
                                                       free_indices_A, free_indices_B,
                                                       isolated_indices_A, isolated_indices_B,
                                                       repeated_indices_A, repeated_indices_B);

    std::unordered_map<int, int64_t> index_extent_map = generate_index_extent_map(min_extent, 4, equal_extents_only, total_unique_indices, unique_indices);

    auto [extents_A, extents_B, extents_C, extents_D] = assign_extents(index_extent_map, nmode_A, idx_A, nmode_B, idx_B, nmode_D, idx_D);

    int outer_nmode_A = subtensor_on_nmode ? nmode_A + rand(1, 4) : nmode_A;
    int outer_nmode_B = subtensor_on_nmode ? nmode_B + rand(1, 4) : nmode_B;
    int outer_nmode_C = subtensor_on_nmode ? nmode_C + rand(1, 4) : nmode_C;
    int outer_nmode_D = subtensor_on_nmode ? nmode_D + rand(1, 4) : nmode_D;

    int* stride_signs_A = choose_stride_signs(nmode_A, negative_strides_enabled, mixed_strides_enabled);
    int* stride_signs_B = choose_stride_signs(nmode_B, negative_strides_enabled, mixed_strides_enabled);
    int* stride_signs_C = choose_stride_signs(nmode_C, negative_strides_enabled, mixed_strides_enabled);
    int* stride_signs_D = choose_stride_signs(nmode_D, negative_strides_enabled, mixed_strides_enabled);

    bool* subtensor_dims_A = choose_subtensor_dims(nmode_A, outer_nmode_A);
    bool* subtensor_dims_B = choose_subtensor_dims(nmode_B, outer_nmode_B);
    bool* subtensor_dims_C = choose_subtensor_dims(nmode_C, outer_nmode_C);
    bool* subtensor_dims_D = choose_subtensor_dims(nmode_D, outer_nmode_D);

    int64_t* outer_extents_A = calculate_outer_extents(outer_nmode_A, extents_A, subtensor_dims_A, subtensor_on_extents);
    int64_t* outer_extents_B = calculate_outer_extents(outer_nmode_B, extents_B, subtensor_dims_B, subtensor_on_extents);
    int64_t* outer_extents_C = calculate_outer_extents(outer_nmode_C, extents_C, subtensor_dims_C, subtensor_on_extents);
    int64_t* outer_extents_D = calculate_outer_extents(outer_nmode_D, extents_D, subtensor_dims_D, subtensor_on_extents);

    int64_t* offsets_A = calculate_offsets(nmode_A, outer_nmode_A, extents_A, outer_extents_A, subtensor_dims_A, subtensor_on_extents);
    int64_t* offsets_B = calculate_offsets(nmode_B, outer_nmode_B, extents_B, outer_extents_B, subtensor_dims_B, subtensor_on_extents);
    int64_t* offsets_C = calculate_offsets(nmode_C, outer_nmode_C, extents_C, outer_extents_C, subtensor_dims_C, subtensor_on_extents);
    int64_t* offsets_D = calculate_offsets(nmode_D, outer_nmode_D, extents_D, outer_extents_D, subtensor_dims_D, subtensor_on_extents);

    int64_t* strides_A = calculate_strides(nmode_A, outer_nmode_A, outer_extents_A, stride_signs_A, subtensor_dims_A);
    int64_t* strides_B = calculate_strides(nmode_B, outer_nmode_B, outer_extents_B, stride_signs_B, subtensor_dims_B);
    int64_t* strides_C = calculate_strides(nmode_C, outer_nmode_C, outer_extents_C, stride_signs_C, subtensor_dims_C);
    int64_t* strides_D = calculate_strides(nmode_D, outer_nmode_D, outer_extents_D, stride_signs_D, subtensor_dims_D);
    
    int64_t size_A = calculate_size(outer_nmode_A, outer_extents_A);
    int64_t size_B = calculate_size(outer_nmode_B, outer_extents_B);
    int64_t size_C = calculate_size(outer_nmode_C, outer_extents_C);
    int64_t size_D = calculate_size(outer_nmode_D, outer_extents_D);

    T* data_A = create_tensor_data<T>(size_A);
    T* data_B = create_tensor_data<T>(size_B);
    T* data_C = create_tensor_data<T>(size_C);
    T* data_D = create_tensor_data<T>(size_D);

    T* A = calculate_tensor_pointer<T>(data_A, nmode_A, extents_A, offsets_A, strides_A);
    T* B = calculate_tensor_pointer<T>(data_B, nmode_B, extents_B, offsets_B, strides_B);
    T* C = calculate_tensor_pointer<T>(data_C, nmode_C, extents_C, offsets_C, strides_C);
    T* D = calculate_tensor_pointer<T>(data_D, nmode_D, extents_D, offsets_D, strides_D);

    T alpha = rand<T>();
    T beta = rand<T>();

    delete[] unique_indices;

    delete[] subtensor_dims_A;
    delete[] subtensor_dims_B;
    delete[] subtensor_dims_C;
    delete[] subtensor_dims_D;

    delete[] outer_extents_A;
    delete[] outer_extents_B;
    delete[] outer_extents_C;
    delete[] outer_extents_D;

    delete[] stride_signs_A;
    delete[] stride_signs_B;
    delete[] stride_signs_C;
    delete[] stride_signs_D;

    delete[] offsets_A;
    delete[] offsets_B;
    delete[] offsets_C;
    delete[] offsets_D;
    
    return {nmode_A, extents_A, strides_A, A, idx_A,
            nmode_B, extents_B, strides_B, B, idx_B,
            nmode_C, extents_C, strides_C, C, idx_C,
            nmode_D, extents_D, strides_D, D, idx_D,
            alpha, beta,
            data_A, data_B, data_C, data_D,
            size_A, size_B, size_C, size_D};
}

// nmode_A, nmode_B, nmode_C, nmode_D, contracted_modes, hadamard_modes, free_indices_A, free_indices_B, isolated_indices_A, isolated_indices_B, repeated_indices_A, repeated_indices_B
// OBS: If something is enabled at least one of those instances will be generated
std::tuple<int, int, int, int,
           int, int, int, int,
           int, int, int, int> generate_index_configuration(int nmode_A, int nmode_B, int nmode_D,
                                                            int contracted_indices, int hadamard_indices,
                                                            bool hadamard_only, bool hadamard_indices_enabled,
                                                            bool isolated_indices_enabled, bool repeated_indices_enabled)
{
    int free_indices_A = 0;
    int free_indices_B = 0;
    int isolated_indices_A = 0;
    int isolated_indices_B = 0;
    int repeated_indices_A = 0;
    int repeated_indices_B = 0;
    if (hadamard_indices == -1 && hadamard_indices_enabled) // If no hadamards defined but are allowed, calculate possible amount of hadamrd indices
    {
        int max_hadamard_indices = nmode_D; // Start with number of modes for D as maximum hadamard indices, maximum possible must be possitive to be valid

        if (nmode_A != -1) // If number of modes for A is defined
        {
            int new_max_hadamard = nmode_A;
            if (contracted_indices != -1)
            {
                new_max_hadamard -= contracted_indices;
            }
            if (isolated_indices_enabled) // A will have at least one isolated index, if enabled, one less available for hadamard
            {
                new_max_hadamard -= 1;
            }
            if (repeated_indices_enabled) // A will have at least one repeated index, if enabled, one less available for hadamard
            {
                new_max_hadamard -= 1;
            }
            if (max_hadamard_indices < 0) // If maximum hadamards is not valid, assign a new value
            {
                max_hadamard_indices = new_max_hadamard;
            }
            else // If maximum hadamards is valid, find the lowest value
            {
                max_hadamard_indices = std::min(max_hadamard_indices, new_max_hadamard); 
            }
        }
        if (nmode_B != -1) // If number of modes for B is defined
        {
            int new_max_hadamard = nmode_B;
            if (contracted_indices != -1)
            {
                new_max_hadamard -= contracted_indices;
            }
            if (isolated_indices_enabled) // B will have at least one isolated index, if enabled, one less available for hadamard
            {
                new_max_hadamard -= 1;
            }
            if (repeated_indices_enabled) // B will have at least one repeated index, if enabled, one less available for hadamard
            {
                new_max_hadamard -= 1;
            }
            if (max_hadamard_indices < 0) // If maximum hadamards is not valid, assign a new value
            {
                max_hadamard_indices = new_max_hadamard;
            }
            else // If maximum hadamards is valid, find the lowest value
            {
                max_hadamard_indices = std::min(max_hadamard_indices, new_max_hadamard); 
            }
        }
        if (nmode_D != -1) // If number of modes for D is defined
        {
            int new_max_hadamard = nmode_D;
            if (contracted_indices != -1)
            {
                new_max_hadamard -= contracted_indices;
            }
            if (max_hadamard_indices < 0) // If maximum hadamards is not valid, assign a new value
            {
                max_hadamard_indices = new_max_hadamard;
            }
            else // If maximum hadamards is valid, find the lowest value
            {
                max_hadamard_indices = std::min(max_hadamard_indices, new_max_hadamard); 
            }
        }

        if (max_hadamard_indices < 0) // If no valid max found, assign a default value
        {
            max_hadamard_indices = 4;
        }

        hadamard_indices = rand(1, max_hadamard_indices);

        if (isolated_indices_enabled == false && repeated_indices_enabled == false)
        {
            if (nmode_A != -1 && nmode_B != -1 && nmode_D != -1)
            {
                if ((nmode_A + nmode_B + nmode_D) % 2 != hadamard_indices % 2)
                {
                    if (hadamard_indices < max_hadamard_indices)
                    {
                        hadamard_indices += 1;
                    }
                    else
                    {
                        hadamard_indices -= 1;
                    }
                }
            }
        }
    }
    else if (hadamard_indices == -1 && hadamard_indices_enabled == false) // No hadamards allowed
    {
        hadamard_indices = 0;
    }

    if (hadamard_only)
    {
        contracted_indices = 0;
    }
    else
    {
        if (contracted_indices == -1)
        {
            if (nmode_A != -1 && nmode_B != -1)
            {
                int max_contracted_indices;
                if (nmode_D != -1)
                {
                    max_contracted_indices = ((nmode_B - hadamard_indices) + (nmode_A - hadamard_indices) - (nmode_D - hadamard_indices))/2;
                }
                else
                {
                    max_contracted_indices = std::min(nmode_A, nmode_B) - hadamard_indices;
                }
                if (isolated_indices_enabled || repeated_indices_enabled)
                {
                    int min_contracted_indices = 0;
                    if (isolated_indices_enabled) // A and B will have at least one isolated index each, if enabled, one less available for contractions
                    {
                        max_contracted_indices -= 1;
                    }
                    if (repeated_indices_enabled) // A and B will have at least one repeated index each, if enabled, one less available for contractions
                    {
                        max_contracted_indices -= 1;
                    }
                    contracted_indices = rand(min_contracted_indices, max_contracted_indices);
                }
                else
                {
                    contracted_indices = max_contracted_indices;
                }
            }
            else if (nmode_A != -1 || nmode_B != -1)
            {
                int min_contracted_indices;
                int max_contracted_indices = std::max(nmode_A, nmode_B) - hadamard_indices; // If one is defined and one is not, the defined one will be more than 0 and the undefined one -1, therefore max will find the defined one
                if (nmode_D != -1)
                {
                    min_contracted_indices = max_contracted_indices - (nmode_D - hadamard_indices);
                }
                else
                {
                    min_contracted_indices = 0;
                }
                if (isolated_indices_enabled) // A and B will have at least one isolated index each, if enabled, one less available for contractions
                {
                    max_contracted_indices -= 1;
                }
                if (repeated_indices_enabled) // A and B will have at least one repeated index each, if enabled, one less available for contractions
                {
                    max_contracted_indices -= 1;
                }
                contracted_indices = rand(min_contracted_indices, max_contracted_indices);
            }
            else // A or B, no constriction on the number of contractions
            {
                contracted_indices = rand(0, 4);
            }
        }
    }

    if (nmode_D == -1)
    {
        nmode_D = hadamard_indices;
        if (hadamard_only == false)
        {
            if (nmode_A != -1 && nmode_B != -1)
            {
                int max_nmode_D = nmode_A + nmode_B - 2 * (contracted_indices + hadamard_indices);
                if (isolated_indices_enabled || repeated_indices_enabled)
                {
                    int min_nmode_D = 0;
                    if (isolated_indices_enabled) // A and B will have at least one isolated index each, if enabled, total of two less free indices for D
                    {
                        max_nmode_D -= 2;
                    }
                    if (repeated_indices_enabled) // A and B will have at least one repeated index each, if enabled, total of two less free indices for D
                    {
                        max_nmode_D -= 2;
                        if (contracted_indices == 0) // If no indices are contracted, see to it that there are two free to allow for repeated indices
                        {
                            min_nmode_D = std::max(min_nmode_D, 2);
                            max_nmode_D = std::max(max_nmode_D, 2);
                        }
                    }
                    nmode_D += rand(min_nmode_D, max_nmode_D);
                }
                else
                {
                    nmode_D += max_nmode_D;
                }
            }
            else if (nmode_A != -1 || nmode_B != -1)
            {
                int min_nmode_D = std::max(nmode_A, nmode_B) - hadamard_indices - contracted_indices;
                int max_nmode_D = std::max(min_nmode_D + 2, 4);
                if (isolated_indices_enabled) // The defined tensor will at least one isolated index each, if enabled, which means that D don't need to assume it to be free
                {
                    min_nmode_D -= 1;
                }
                if (repeated_indices_enabled) // The defined tensor will at least one repeated index each, if enabled, which means that D don't need to assume it to be free
                {
                    min_nmode_D -= 1;
                    if (contracted_indices == 0) // If no indices are contracted, see to it that there are two free to allow for repeated indices
                    {
                        min_nmode_D = std::max(min_nmode_D, 2);
                        max_nmode_D = std::max(max_nmode_D, 2);
                    }
                }
                nmode_D += rand(min_nmode_D, max_nmode_D);
            }
            else
            {
                if (repeated_indices_enabled && contracted_indices == 0) // If no indices are contracted, see to it that there are two free to allow for repeated indices
                {
                    nmode_D += std::max(rand(0, 4), 2);
                }
                else
                {
                    nmode_D += rand(0, 4);
                }
            }
        }
    }

    if (nmode_A == -1) // If no number of modes defined for A
    {
        isolated_indices_A = isolated_indices_enabled ? rand(1, 4) : 0; // Pick a random amount of isolated indices, if allowed
        repeated_indices_A = repeated_indices_enabled ? rand(1, 4) : 0; // Pick a random amount of repeated indices, if allowed
        nmode_A = isolated_indices_A + repeated_indices_A + hadamard_indices + contracted_indices; // Assign all known number of indices
        if (nmode_B != -1) // If B, D and the number of contracted indices are defined, A needs to follow those constraints
        {
            if (isolated_indices_enabled || repeated_indices_enabled)
            {
                int min_free_indices = nmode_D - (nmode_B - contracted_indices); // Minimum is the amount of needed to fill D with B exausted
                int max_free_indices = nmode_D - hadamard_indices; // D is only indices from A
                if (isolated_indices_enabled) // B will at least one isolated index each, if enabled, which means one less to accomodate for D, A must have more free indices
                {
                    min_free_indices += 1;
                }
                if (repeated_indices_enabled) // B will at least one repeated index each, if enabled, which means one less to accomodate for D, A must have more free indices
                {
                    min_free_indices += 1;
                    if (contracted_indices == 0) // If no indices are contracted, leave at least one free index to tensor B
                    {
                        max_free_indices = std::max(min_free_indices, max_free_indices - 1);
                    }
                }
                min_free_indices = std::max(0, min_free_indices); // Make sure free indices can't be negative
                free_indices_A = rand(min_free_indices, max_free_indices);
            }
            else
            {
                free_indices_A = nmode_D - (nmode_B - contracted_indices);
            }
        }
        else
        {
            int min_free_indices = 0;
            int max_free_indices = nmode_D - hadamard_indices;
            if (repeated_indices_enabled && contracted_indices == 0) // If no indices are contracted and there are repeated indices, A needs at least one free index, leave at least one free index to tensor B
            {
                min_free_indices = 1;
                max_free_indices = std::max(min_free_indices, max_free_indices - 1);
            }
            free_indices_A = rand(min_free_indices, max_free_indices);
        }
        nmode_A += free_indices_A;
    }
    else
    {
        if (isolated_indices_enabled || repeated_indices_enabled)
        {
            int min_free_indices = 0;
            int max_free_indices = std::min(nmode_D, nmode_A - hadamard_indices - contracted_indices);
            if (isolated_indices_enabled) 
            {
                max_free_indices -= 1; // A will have at least one isolated index, if enabled, one less available to accomodate for D
            }
            if (repeated_indices_enabled) 
            {
                max_free_indices -= 1; // A will have at least one repeated index, if enabled, one less available to accomodate for D
            }
            if (nmode_B != -1)
            {
                min_free_indices = nmode_D - (nmode_B - contracted_indices);
                if (isolated_indices_enabled) 
                {
                    min_free_indices += 1; // B will have at least one isolated index, if enabled, one less available to accomodate for D
                }
                if (repeated_indices_enabled) 
                {
                    min_free_indices += 1; // B will have at least one isolated index, if enabled, one less available to accomodate for D
                }
            }
            free_indices_A = rand(min_free_indices, max_free_indices);
            if (isolated_indices_enabled) 
            {
                int min_repeated_indices = repeated_indices_enabled ? 1 : 0; // If enabled, make sure to reserve at least one index for repeated indices
                isolated_indices_A = rand(1, nmode_A - free_indices_A - hadamard_indices - contracted_indices - min_repeated_indices); // Pick an amount of isolated indices from available space
            }
            if (repeated_indices_enabled)
            {
                repeated_indices_A = nmode_A - free_indices_A - hadamard_indices - contracted_indices - isolated_indices_A; // Repeated indices gets what's left
            }
        }
        else
        {
            free_indices_A = nmode_A - hadamard_indices - contracted_indices;
        }
    }

    if (nmode_B == -1) // If no number of modes defined for B
    {
        isolated_indices_B = isolated_indices_enabled ? rand(1, 4) : 0; // Pick a random amount of isolated indices, if allowed
        repeated_indices_B = repeated_indices_enabled ? rand(1, 4) : 0; // Pick a random amount of repeated indices, if allowed
        free_indices_B = nmode_D - hadamard_indices - free_indices_A;
        nmode_B = isolated_indices_B + repeated_indices_B + hadamard_indices + contracted_indices + free_indices_B;
    }
    else
    {
        free_indices_B = nmode_D - hadamard_indices - free_indices_A;
        if (isolated_indices_enabled) 
        {
            int min_repeated_indices = repeated_indices_enabled ? 1 : 0; // If enabled, make sure to reserve at least one index for repeated indices
            isolated_indices_B = rand(1, nmode_B - free_indices_B - hadamard_indices - contracted_indices - min_repeated_indices); // Pick an amount of isolated indices from available space
        }
        if (repeated_indices_enabled)
        {
            repeated_indices_B = nmode_B - free_indices_B - hadamard_indices - contracted_indices - isolated_indices_B; // Repeated indices gets what's left
        }
    }

    return {nmode_A, nmode_B, nmode_D, nmode_D, contracted_indices, hadamard_indices, free_indices_A, free_indices_B, isolated_indices_A, isolated_indices_B, repeated_indices_A, repeated_indices_B};
}

int* generate_unique_indices(int64_t total_unique_indices)
{
    int* unique_indices = new int[total_unique_indices];
    for (int i = 0; i < total_unique_indices; i++)
    {
        unique_indices[i] = 'a' + i;
    }
    std::shuffle(unique_indices, unique_indices + total_unique_indices, rand_engine()); // Shuffle the unique indices
    return unique_indices;
}

std::tuple<int64_t*, int64_t*, int64_t*, int64_t*> assign_indices(int* unique_indices,
                                                                  int contracted_indices, int hadamard_indices,
                                                                  int free_indices_A, int free_indices_B,
                                                                  int isolated_indices_A, int isolated_indices_B,
                                                                  int repeated_indices_A, int repeated_indices_B)
{
    // Create index arrays
    int64_t* idx_A = new int64_t[repeated_indices_A + isolated_indices_A + free_indices_A + hadamard_indices + contracted_indices];
    int64_t* idx_B = new int64_t[repeated_indices_B + isolated_indices_B + free_indices_B + hadamard_indices + contracted_indices];
    int64_t* idx_C = new int64_t[free_indices_A + hadamard_indices + free_indices_B];
    int64_t* idx_D = new int64_t[free_indices_A + hadamard_indices + free_indices_B];

    /*
     * Intended layout of indices:
     *  isolated_indices_A - free_indices_A - hadamard_indices - free_indices_B - isolated_indices_B - contracted_indices
     * |---------------------idx_A---------------------|                                            |-----idx_A------|
     *                                       |-----------------------------idx_B-------------------------------------|
     *                      |---------------------idx_C----------------------|
     */

    // Copy indices into each index array
    std::copy(unique_indices, unique_indices + isolated_indices_A + free_indices_A + hadamard_indices, idx_A); // Assign indices to A

    std::copy(unique_indices + isolated_indices_A + free_indices_A + hadamard_indices + free_indices_B + isolated_indices_B,
              unique_indices + isolated_indices_A + free_indices_A + hadamard_indices + free_indices_B + isolated_indices_B + contracted_indices,
              idx_A + isolated_indices_A + free_indices_A + hadamard_indices); // Needs a second copy for contractions

    std::copy(unique_indices + isolated_indices_A + free_indices_A,
              unique_indices + isolated_indices_A + free_indices_A + hadamard_indices + free_indices_B + isolated_indices_B + contracted_indices,
              idx_B); // Assign indices to B

    std::copy(unique_indices + isolated_indices_A,
              unique_indices + isolated_indices_A + free_indices_A + hadamard_indices + free_indices_B,
              idx_D); // Assign indices to D

    std::shuffle(idx_D, idx_D + (free_indices_A + hadamard_indices + free_indices_B), rand_engine()); // Shuffle indices for D

    std::copy(idx_D,
              idx_D + free_indices_A + hadamard_indices + free_indices_B,
              idx_C); // C has the same indices as D

    for (int i = 0; i < repeated_indices_A; i++) // Add repeated indices to A
    {
        idx_A[i + isolated_indices_A + free_indices_A + hadamard_indices + contracted_indices] = idx_A[rand(0, isolated_indices_A + free_indices_A + hadamard_indices + contracted_indices - 1)];
    }

    for (int i = 0; i < repeated_indices_B; i++) // Add repeated indices to B
    {
        idx_B[i + isolated_indices_B + free_indices_B + hadamard_indices + contracted_indices] = idx_B[rand(0, isolated_indices_B + free_indices_B + hadamard_indices + contracted_indices - 1)];
    }

    std::shuffle(idx_A, idx_A + repeated_indices_A + isolated_indices_A + free_indices_A + hadamard_indices + contracted_indices, rand_engine()); // Shuffle final indices for A

    std::shuffle(idx_B, idx_B + repeated_indices_B + isolated_indices_B + free_indices_B + hadamard_indices + contracted_indices, rand_engine()); // Shuffle final indices for B
    
    return {idx_A, idx_B, idx_C, idx_D};
}

std::unordered_map<int, int64_t> generate_index_extent_map(int64_t min_extent, int64_t max_extent,
                                                           bool equal_extents_only,
                                                           int64_t total_unique_indices, int* unique_indices)
{
    std::unordered_map<int, int64_t> index_to_extent;
    int extent = rand(min_extent, max_extent);
    for (int64_t i = 0; i < total_unique_indices; i++)
    {
        if (!equal_extents_only) extent = rand(min_extent, max_extent);
        index_to_extent[unique_indices[i]] = extent;
    }
    return index_to_extent;
}

std::tuple<int64_t*, int64_t*, int64_t*, int64_t*> assign_extents(std::unordered_map<int, int64_t> index_extent_map,
                                                                  int nmode_A, int64_t* idx_A,
                                                                  int nmode_B, int64_t* idx_B,
                                                                  int nmode_D, int64_t* idx_D)
{
    // Create extent arrays
    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_C = new int64_t[nmode_D];
    int64_t* extents_D = new int64_t[nmode_D];

    // Map extents to tensors based on their indices
    for (int64_t i = 0; i < nmode_A; i++) // Assign extents to A
    {
        extents_A[i] = index_extent_map[idx_A[i]];
    }
    for (int64_t i = 0; i < nmode_B; i++) // Assign extents to B
    {
        extents_B[i] = index_extent_map[idx_B[i]]; // Assign extents to B
    }
    for (int64_t i = 0; i < nmode_D; i++)
    {
        extents_D[i] = index_extent_map[idx_D[i]]; // Assign extents to D
    }

    std::copy(extents_D, extents_D + nmode_D, extents_C);

    return {extents_A, extents_B, extents_C, extents_D};
}

int* choose_stride_signs(int nmode, bool negative_strides_enabled, bool mixed_strides_enabled)
{
    int* stride_signs = new int[nmode];

    for (size_t i = 0; i < nmode; i++)
    {
        if ((negative_strides_enabled && !mixed_strides_enabled) || (rand(0, 1) == 0 && negative_strides_enabled && mixed_strides_enabled))
        {
            stride_signs[i] = -1;
        }
        else
        {
            stride_signs[i] = 1;
        }
    }
    return stride_signs;
}

bool* choose_subtensor_dims(int nmode, int outer_nmode)
{
    bool* subtensor_dims = new bool[outer_nmode];
    int idx = 0;
    for (int i = 0; i < outer_nmode; i++)
    {
        if ((rand((float)0, (float)1) < (float)nmode/(float)outer_nmode || outer_nmode - i == nmode - idx) && nmode - idx > 0)
        {
            subtensor_dims[i] = true;
            idx++;
        }
        else
        {
            subtensor_dims[i] = false;
        }
    }
    return subtensor_dims;
}

int64_t* calculate_outer_extents(int outer_nmode, int64_t* extents, bool* subtensor_dims, bool lower_extents)
{
    int64_t* outer_extents = new int64_t[outer_nmode];
    int idx = 0;
    for (int i = 0; i < outer_nmode; i++)
    {
        if (subtensor_dims[i])
        {
            int extension = rand(1, 4);
            outer_extents[i] = lower_extents ? extents[idx] + extension : extents[idx];
            idx++;
        }
        else
        {
            outer_extents[i] = lower_extents ? rand(1, 8) : rand(1, 4);
        }
    }
    return outer_extents;
}

int64_t* calculate_offsets(int nmode, int outer_nmode, int64_t* extents, int64_t* outer_extents, bool* subtensor_dims, bool lower_extents)
{
    int64_t* offsets = new int64_t[nmode];
    int idx = 0;
    for (int i = 0; i < outer_nmode; i++)
    {
        if (subtensor_dims[i])
        {
            offsets[idx] = lower_extents && outer_extents[i] - extents[idx] > 0 ? rand((int64_t)0, outer_extents[i] - extents[idx]) : 0;
            idx++;
        }
    }
    return offsets;
}

int64_t* calculate_strides(int nmode, int outer_nmode, int64_t* outer_extents, int* stride_signs, bool* subtensor_dims)
{
    int64_t* strides = new int64_t[nmode];
    int64_t str = 1;
    int idx = 0;
    for (int i = 0; i < outer_nmode; i++)
    {
        if (subtensor_dims[i])
        {
            strides[idx] = str * stride_signs[idx];
            str *= outer_extents[i];
            idx++;
        }
        else
        {
            str *= outer_extents[i];
        }
    }
    return strides;
}

int64_t* calculate_strides(int nmode, int64_t* extents)
{
    int64_t * strides = new int64_t[nmode];
    for (size_t i = 0; i < nmode; i++)
    {
        strides[i] = i == 0 ? 1 : strides[i - 1] * extents[i - 1];
    }
    return strides;
}

int calculate_size(int nmode, int64_t* extents)
{
    int size = 1;
    for (size_t i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    return size;
}

template<typename T>
T* create_tensor_data(int64_t size)
{
    T* data = new T[size];
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand<T>();
    }
    return data;
}

template<typename T>
T* create_tensor_data(int64_t size, T* min_value, T* max_value)
{
    T* data = new T[size];
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand<T>(min_value, max_value);
    }
    return data;
}

template<typename T>
T* calculate_tensor_pointer(T* pointer, int nmode, int64_t* extents, int64_t* offsets, int64_t* strides)
{
    T* new_pointer = pointer;

    for (int i = 0; i < nmode; i++)
    {
        if (strides[i] < 0)
        {
            new_pointer -= (extents[i] - 1) * strides[i];
            new_pointer -= offsets[i] * strides[i];
        }
        else {
            new_pointer += offsets[i] * strides[i];
        }
    }
    return new_pointer;
}

void* calculate_tensor_pointer(void* pointer, int nmode, int64_t* extents, int64_t* offsets, int64_t* strides, unsigned long data_size)
{
    intptr_t new_pointer = (intptr_t)pointer;

    for (int i = 0; i < nmode; i++)
    {
        if (strides[i] < 0)
        {
            new_pointer -= (extents[i] - 1) * strides[i] * data_size;
            new_pointer -= offsets[i] * strides[i] * data_size;
        }
        else {
            new_pointer += offsets[i] * strides[i] * data_size;
        }
    }
    return (void*)new_pointer;
}

template<typename T>
std::tuple<T*, T*> copy_tensor_data(int64_t size, T* data, T* pointer)
{
    T* new_data = new T[size];
    std::copy(data, data + size, new_data);
    T* new_pointer = (T*)((intptr_t)new_data + (intptr_t)pointer - (intptr_t)data);
    return {new_pointer, new_data};
}

template<typename T>
T* copy_tensor_data(int64_t size, T* data)
{
    T* new_data = new T[size];
    std::copy(data, data + size, new_data);
    return new_data;
}

int calculate_tensor_size(int nmode, int* extents)
{
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    return size;
}

template<typename T>
T rand(T min, T max)
{
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rand_engine());
    }
    else if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(rand_engine());
    }
    else if constexpr (is_complex_v<T>) {
        using value_type = typename T::value_type;

        std::uniform_real_distribution<value_type> dist_real(
            min.real(), max.real()
        );
        std::uniform_real_distribution<value_type> dist_imag(
            min.imag(), max.imag()
        );

        return T{
            dist_real(rand_engine()),
            dist_imag(rand_engine())
        };
    }
}

template<typename T>
T rand()
{
    return rand<T>(-std::numeric_limits<T>::max(), std::numeric_limits<T>::max());
}

template<typename T>
T random_choice(int size, T* choices)
{
    return choices[rand(0, size - 1)];
}

char* swap_indices(char* indices, int nmode_A, int nmode_B, int nmode_D)
{
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

void rotate_indices(int64_t* idx, int nmode, int64_t* extents, int64_t* strides)
{
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

void increment_coordinates(int64_t* coordinates, int nmode, int64_t* extents)
{
    if (nmode <= 0)
    {
        return;
    }

    int k = 0;
    do
    {
        coordinates[k] = (coordinates[k] + 1) % extents[k];
        k++;
    } while (coordinates[k - 1] == 0 && k < nmode);
}

void print_tensor(int nmode, int64_t* extents, int64_t* strides)
{
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
}

template<typename T>
void print_tensor(int nmode, int64_t* extents, int64_t* strides, T* data)
{
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

void add_incorrect_idx(int64_t max_idx, int* nmode, int64_t** idx, int64_t** extents, int64_t** strides)
{
    int nmode_tmp = *nmode + rand(1, 5);
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

void add_idx(int* nmode, int64_t** idx, int64_t** extents, int64_t** strides, int64_t additional_idx, int64_t additional_extents, int64_t additional_strides)
{
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

bool test_hadamard_product()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, false, false, false, false, true, true);

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    int op_A = 0;
    int op_B = 0;
    int op_C = 0;
    int op_D = 0;

    TAPP_tensor_product plan;
    TAPP_handle handle;
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(D, E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    delete[] extents_A;
    delete[] strides_A;
    delete[] extents_B;
    delete[] strides_B;
    delete[] extents_C;
    delete[] strides_C;
    delete[] extents_D;
    delete[] strides_D;
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

bool test_contraction()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>();

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_commutativity()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>();

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

    auto [F, data_F] = copy_tensor_data(size_D, data_D, D);

    auto [G, data_G] = copy_tensor_data(size_D, data_D, D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
    TAPP_create_handle(&handle);
    TAPP_tensor_product planAB;
    TAPP_create_tensor_product(&planAB, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_tensor_product planBA;
    TAPP_create_tensor_product(&planBA, handle, 0, info_B, idx_B, 0, info_A, idx_A, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(planAB, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    TAPP_execute_product(planBA, exec, &status, (void*)&alpha, (void*)B, (void*)A, (void*)&beta, (void*)C, (void*)F);

    run_tblis_mult(nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, G, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D) && compare_tensors(data_F, data_G, size_D) && compare_tensors(data_D, data_F, size_D);
    
    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(planAB);
    TAPP_destroy_tensor_product(planBA);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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
    delete[] data_F;
    delete[] data_G;

    return result;
}

bool test_permutations()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, rand(2, 4));
          
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    TAPP_create_handle(&handle);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);
    
    bool result = true;

    for (int i = 0; i < nmode_D; i++)
    {
        TAPP_tensor_info info_C;
        TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
        TAPP_tensor_info info_D;
        TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);
        TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
        TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);
        run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                    nmode_B, extents_B, strides_B, B, 0, idx_B,
                    nmode_C, extents_C, strides_C, C, 0, idx_D,
                    nmode_D, extents_D, strides_D, E, 0, idx_D,
                    alpha, beta);
        
        result = result && compare_tensors(data_D, data_E, size_D);

        rotate_indices(idx_C, nmode_C, extents_C, strides_C);
        rotate_indices(idx_D, nmode_D, extents_D, strides_D);
        TAPP_destroy_tensor_info(info_C);
        TAPP_destroy_tensor_info(info_D);
        TAPP_destroy_tensor_product(plan);
    }
    
    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
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

    return result;
}

bool test_equal_extents()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_outer_product()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, 0);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;
    
    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_full_contraction()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, 0);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_zero_dim_tensor_contraction()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(0);//2,2,0,2);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_one_dim_tensor_contraction()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(1);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_subtensor_unchanged_nmode()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_subtensor_lower_nmode()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, true, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_negative_strides()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, false, false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);
    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);    

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return true;
}

bool test_negative_strides_subtensor_unchanged_nmode()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, true, false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_negative_strides_subtensor_lower_nmode()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, true, true, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_mixed_strides()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, false, false, false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);
    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return true;
}

bool test_mixed_strides_subtensor_unchanged_nmode()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, true, false, false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_mixed_strides_subtensor_lower_nmode()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, true, true, false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, D);
    
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_contraction_double_precision()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<double>();

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_contraction_complex()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<std::complex<float>>();

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C32, nmode_D, extents_D, strides_D);

    int op_A = rand(0, 1);
    int op_B = rand(0, 1);
    int op_C = rand(0, 1);
    int op_D = rand(0, 1);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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
        
    return result;
}

bool test_contraction_complex_double_precision()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<std::complex<double>>(2,2,0,2);//2,2,0,2);

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C64, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C64, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C64, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C64, nmode_D, extents_D, strides_D);

    int op_A = rand(0, 1);
    int op_B = rand(0, 1);
    int op_C = rand(0, 1);
    int op_D = rand(0, 1);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    int terr = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                     nmode_B, extents_B, strides_B, B, op_B, idx_B,
                     nmode_C, extents_C, strides_C, C, op_C, idx_D,
                     nmode_D, extents_D, strides_D, E, op_D, idx_D,
                     alpha, beta);
    // std::complex<double> zma = 1.0+1.0e-12;
    // data_D[0] = data_D[0]*zma;
    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_zero_stride()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, rand(1, 4));

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

    if (nmode_A > 0)
    {
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_isolated_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, false, false, false, false, false, false, false, true);

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_repeated_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, -1, -1, 1, false, false, false, false, false, false, false, true);

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_hadamard_and_free()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, -1, 0, -1, 1, false, false, false, false, false, true);

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)data_A, (void*)data_B, (void*)&beta, (void*)data_C, (void*)data_D);

    run_tblis_mult(nmode_A, extents_A, strides_A, data_A, 0, idx_A,
                   nmode_B, extents_B, strides_B, data_B, 0, idx_B,
                   nmode_C, extents_C, strides_C, data_C, 0, idx_D,
                   nmode_D, extents_D, strides_D, data_E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_hadamard_and_contraction()
{
    int input_nmode = rand(0, 4);
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, input_nmode, -1, input_nmode, 1, false, false, false, false, false, true);

    auto [E, data_E] = copy_tensor_data(size_D, data_D, D);

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)data_A, (void*)data_B, (void*)&beta, (void*)data_C, (void*)data_D);

    run_tblis_mult(nmode_A, extents_A, strides_A, data_A, 0, idx_A,
                   nmode_B, extents_B, strides_B, data_B, 0, idx_B,
                   nmode_C, extents_C, strides_C, data_C, 0, idx_D,
                   nmode_D, extents_D, strides_D, data_E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return result;
}

bool test_error_too_many_idx_D()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>();

    int64_t max_idx = 0;
    for (size_t i = 0; i < nmode_A; i++)
    {
        if (max_idx < idx_A[i])
        {
            max_idx = idx_A[i];
        }
    }
    for (size_t i = 0; i < nmode_B; i++)
    {
        if (max_idx < idx_B[i])
        {
            max_idx = idx_B[i];
        }
    }
    for (size_t i = 0; i < nmode_D; i++)
    {
        if (max_idx < idx_D[i])
        {
            max_idx = idx_D[i];
        }
    }

    add_incorrect_idx(max_idx, &nmode_D, &idx_D, &extents_D, &strides_D);

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return error_status == 7;
}

bool test_error_non_matching_ext()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, rand(1, 4));
    
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
        random_index = rand(0, nmode_A - 1);
        extents_A[random_index] += rand(1, 5);
        break;
    case 1:
        random_index = rand(0, nmode_B - 1);
        extents_B[random_index] += rand(1, 5);
        break;
    case 2:
        random_index = rand(0, nmode_D - 1);
        extents_D[random_index] += rand(1, 5);
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return error_status == 1 || error_status == 2 || error_status == 3;
}

bool test_error_C_other_structure()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, rand(1, 4));

    int64_t max_idx = 0;
    for (size_t i = 0; i < nmode_C; i++)
    {
        if (max_idx < idx_C[i])
        {
            max_idx = idx_C[i];
        }
    }

    int random_error = rand(0, 2);
    int random_index = 0;

    switch (random_error)
    {
    case 0:
        add_incorrect_idx(max_idx, &nmode_C, &idx_C, &extents_C, &strides_C);
        break;
    case 1:
        if (nmode_C > 1)
        {
            random_index = rand(0, nmode_C - 1);
            idx_C[random_index] = random_index == 0 ? idx_C[random_index + 1] : idx_C[random_index - 1];
        }
        else {
            add_idx(&nmode_C, &idx_C, &extents_C, &strides_C, idx_C[0], extents_C[0], strides_C[0]);
        }
        break;
    case 2:
        random_index = nmode_C == 1 ? 0 : rand(0, nmode_C - 1);
        extents_C[random_index] += rand(1, 5);
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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return error_status == 5 || error_status == 6 || error_status == 7;
}

bool test_error_aliasing_within_D()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_pseudorandom_contraction<float>(-1, -1, rand(2, 4), -1, -1, 2);

    int scewed_index = rand(1, nmode_D - 1);
    int signs[2] = {-1, 1};
    strides_D[scewed_index] = random_choice(2, signs) * (strides_D[scewed_index - 1] * extents_D[scewed_index - 1] - rand((int64_t)1, strides_D[scewed_index - 1] * extents_D[scewed_index - 1] - 1));

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
    TAPP_create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    TAPP_create_executor(&exec);

    int error_status = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

    return error_status == 8;
}
