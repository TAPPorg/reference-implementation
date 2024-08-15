/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#include <iostream>
#include <random>
#include <tuple>
#include <string>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "tblis.h"
#pragma GCC diagnostic pop
extern "C" {
    #include "tapp.h"
}

void run_tblis_mult(int nmode_A, int* extents_A, int* strides_A, float* A, int op_A, char* idx_A,
                    int nmode_B, int* extents_B, int* strides_B, float* B, int op_B, char* idx_B,
                    int nmode_C, int* extents_C, int* strides_C, float* C, int op_C, char* idx_C,
                    int nmode_D, int* extents_D, int* strides_D, float* D, int op_D, char* idx_D,
                    float alpha, float beta);

bool compare_tensors(float* A, float* B, int size);

std::tuple<int, int64_t*, int64_t*, float*, int, int64_t*,
           int, int64_t*, int64_t*, float*, int, int64_t*,
           int, int64_t*, int64_t*, float*, int, int64_t*,
           int, int64_t*, int64_t*, float*, int, int64_t*,
           float, float,
           float*, float*, float*, float*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction(int nmode_A, int nmode_B, int nmode_D, 
                                                                       int contractions, bool equal_extents,
                                                                       bool lower_extents, bool lower_idx,
                                                                       bool negative_str);

std::string str(bool b);

int randi(int min, int max);

float randf(float min, float max);

char* swap_indices(char* indices, int IDXA, int IDXB, int IDXD);

void print_tensor(int nmode, int64_t* extents, int64_t* strides, float* data);

std::tuple<float*, float*> copy_tensor_data(int64_t size, float* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str);

std::tuple<float*, float*> zero_tensor(int size, int IDX, int* offset, int* STR, bool negative_str);

float* zero_tensor(int IDX, int* EXT);

float* copy_tensor_data(int IDX, int64_t* EXT, float* data);

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
    return 0;
}

void run_tblis_mult(int nmode_A, int64_t* extents_A, int64_t* strides_A, float* A, int op_A, int64_t* idx_A,
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

bool compare_tensors(float* A, float* B, int size) {
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        float rel_diff = fabsf((A[i] - B[i]) / (A[i] > B[i] ? A[i] : B[i]));
        if (rel_diff > 0.000005)
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << rel_diff << std::endl;
            found = true;
        }
    }
    return !found;
}

std::tuple<int, int64_t*, int64_t*, float*, int, int64_t*,
           int, int64_t*, int64_t*, float*, int, int64_t*,
           int, int64_t*, int64_t*, float*, int, int64_t*,
           int, int64_t*, int64_t*, float*, int, int64_t*,
           float, float,
           float*, float*, float*, float*,
           int64_t, int64_t, int64_t, int64_t,
           int64_t*, int64_t*, int64_t*, int64_t*> generate_contraction(int nmode_A = -1, int nmode_B = -1,
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
        data_A[i] = randf(0, 1);
    }
    for (int i = 0; i < size_B; i++)
    {
        data_B[i] = randf(0, 1);
    }
    for (int i = 0; i < size_C; i++)
    {
        data_C[i] = randf(0, 1);
    }
    for (int i = 0; i < size_D; i++)
    {
        data_D[i] = randf(0, 1);
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

    float alpha = randf(0, 1);
    float beta = randf(0, 1);

    int op_A = 0;
    int op_B = 0;
    int op_C = 0;
    int op_D = 0;

    return {nmode_A, extents_A, strides_A, A, op_A, idx_A,
            nmode_B, extents_B, strides_B, B, op_B, idx_B,
            nmode_C, extents_C, strides_C, C, op_C, idx_C,
            nmode_D, extents_D, strides_D, D, op_D, idx_D,
            alpha, beta,
            data_A, data_B, data_C, data_D,
            size_A, size_B, size_C, size_D,
            offset_A, offset_B, offset_C, offset_D};
}

std::tuple<float*, float*> copy_tensor_data(int64_t size, float* data, int nmode, int64_t* offset, int64_t* strides, bool negative_str = false) {
    float* dataA = new float[size];
    std::copy(data, data + size, dataA);
    float* A = negative_str ? dataA + size - 1 : dataA;
    for (int i = 0; i < nmode; i++)
    {
        A += offset[i] * strides[i];
    }
    return {A, dataA};
}

std::tuple<float*, float*> zero_tensor(int size, int nmode, int* offset, int* strides, bool negative_str = false) {
    float* dataA = new float[size];
    for (int i = 0; i < size; i++)
    {
        dataA[i] = 0;
    }
    float* A = negative_str ? dataA + size - 1 : dataA;
    for (int i = 0; i < nmode; i++)
    {
        A += offset[i] * strides[i];
    }
    return {A, dataA};
}

float* copy_tensor_data(int nmode, int64_t* extents, float* data) {
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    float* copy = new float[size];
    for (int i = 0; i < size; i++)
    {
        copy[i] = data[i];
    }
    return copy;
}

float* zero_tensor(int nmode, int* extents) {
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        size *= extents[i];
    }
    float* zero = new float[size];
    for (int i = 0; i < size; i++)
    {
        zero[i] = 0;
    }
    return zero;
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

char* swap_indices(char* EINSUM, int IDXA, int IDXB, int IDXD) {
    char* swapped = new char[IDXA + IDXB + IDXD + 7];
    for (int i = 0; i < IDXB; i++)
    {
        swapped[i] = EINSUM[IDXA + 2 + i];
    }
    swapped[IDXB] = ',';
    swapped[IDXB+1] = ' ';
    for (int i = 0; i < IDXA; i++)
    {
        swapped[i + IDXB + 2] = EINSUM[i];
    }
    swapped[IDXA+IDXB+2] = ' ';
    swapped[IDXA+IDXB+3] = '-';
    swapped[IDXA+IDXB+4] = '>';
    swapped[IDXA+IDXB+5] = ' ';
    for (int i = 0; i < IDXD; i++)
    {
        swapped[i + IDXB + IDXA + 6] = EINSUM[IDXA + IDXB + 6 + i];
    }
    swapped[IDXA+IDXB+IDXD+6] = '\0';
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

void print_tensor(int nmode, int64_t* extents, int64_t* strides, float* data) {
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

    float* E = copy_tensor_data(nmode, extents, D);

    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode, extents, strides);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode, extents, strides);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode, extents, strides);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode, extents, strides);

    int op_A = 0;
    int op_B = 0;
    int op_C = 0;
    int op_D = 0;

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode, extents, strides, A, op_A, idx_A,
                   nmode, extents, strides, B, op_B, idx_B,
                   nmode, extents, strides, C, op_C, idx_D,
                   nmode, extents, strides, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(D, E, size);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction();

    auto [E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction();

    auto [E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);

    auto [F, data_F] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);

    auto [G, data_G] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);

    auto [C2, data_C2] = copy_tensor_data(size_C, data_C, nmode_C, offset_C, strides_C);

    auto [C3, data_C3] = copy_tensor_data(size_C, data_C, nmode_C, offset_C, strides_C);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_handle handle = 0;
    TAPP_tensor_product planAB = 0;
    TAPP_create_tensor_product(&planAB, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_tensor_product planBA = 0;
    TAPP_create_tensor_product(&planBA, handle, op_B, info_B, idx_B, op_A, info_A, idx_A, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(planAB, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C2, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);
    
    TAPP_execute_product(planBA, 0, &status, (void*)&alpha, (void*)B, (void*)A, (void*)&beta, (void*)C, (void*)F);

    run_tblis_mult(nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_C, extents_C, strides_C, C3, op_C, idx_D,
                   nmode_D, extents_D, strides_D, G, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D) && compare_tensors(data_F, data_G, size_D) && compare_tensors(data_D, data_F, size_D);
    
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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction();
          
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_status status = 0;
    
    bool result = true;

    for (int i = 0; i < nmode_D; i++)
    {
        auto [C2, copy_C2] = copy_tensor_data(size_C, data_C, nmode_C, offset_C, strides_C);
        TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
        TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);
        run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                    nmode_B, extents_B, strides_B, B, op_B, idx_B,
                    nmode_C, extents_C, strides_C, C2, op_C, idx_D,
                    nmode_D, extents_D, strides_D, E, op_D, idx_D,
                    alpha, beta);
        result = result && compare_tensors(data_D, data_E, size_D);
        rotate_indices(idx_D, nmode_D, extents_D, strides_D);
        rotate_indices(idx_C, nmode_C, extents_C, strides_C);
        TAPP_destory_tensor_product(plan);
        delete[] copy_C2;
    }
    
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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);

    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(-1, -1, randi(0, 4), 0);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(-1, -1, 0);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(0);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(1);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(-1, -1, randi(0, 4), randi(0, 4), false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(-1, -1, randi(0, 4), randi(0, 4), false, true, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(-1, -1, randi(0, 4), randi(0, 4), false, false, false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D, true);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(-1, -1, randi(0, 4), randi(0, 4), false, true, false, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D, true);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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
    auto [nmode_A, extents_A, strides_A, A, op_A, idx_A,
          nmode_B, extents_B, strides_B, B, op_B, idx_B,
          nmode_C, extents_C, strides_C, C, op_C, idx_C,
          nmode_D, extents_D, strides_D, D, op_D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D,
          offset_A, offset_B, offset_C, offset_D] = generate_contraction(-1, -1, randi(0, 4), randi(0, 4), false, true, true, true);
    
    auto[E, data_E] = copy_tensor_data(size_D, data_D, nmode_D, offset_D, strides_D, true);
    
    TAPP_tensor_info info_A = 0;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B = 0;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);
    TAPP_tensor_info info_C = 0;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
    TAPP_tensor_info info_D = 0;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan = 0;
    TAPP_handle handle = 0;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status = 0;

    TAPP_execute_product(plan, 0, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C, op_C, idx_D,
                   nmode_D, extents_D, strides_D, E, op_D, idx_D,
                   alpha, beta);

    bool result = compare_tensors(data_D, data_E, size_D);

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