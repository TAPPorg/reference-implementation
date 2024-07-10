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
    #include "product.h"
}

void run_tblis_mult(int IDXA, int* EXTA, int* STRA, float* A,
                    int IDXB, int* EXTB, int* STRB, float* B,
                    int IDXC, int* EXTC, int* STRC, float* C,
                    int IDXD, int* EXTD, int* STRD, float* D,
                    float ALPHA, float BETA, bool FA, bool FB, bool FC, char* EINSUM);

bool compare_tensors(float* A, float* B, int size);

std::tuple<int, int*, int*, float*,
           int, int*, int*, float*,
           int, int*, int*, float*,
           int, int*, int*, float*,
           float, float, bool, bool, bool, char*,
           float*, float*, float*, float*,
           int, int, int, int,
           int*, int*, int*, int*> generate_contraction(int IDXA, int IDXB, int IDXD, 
                                                                       int contractions, bool equal_extents,
                                                                       bool lower_extents, bool lower_idx,
                                                                       bool negative_str);

std::string str(bool b);

int randi(int min, int max);

float randf(float min, float max);

char* swap_indices(char* indices, int IDXA, int IDXB, int IDXD);

void print_tensor(int IDX, int* EXT, int* STR, float* data);

std::tuple<float*, float*> copy_tensor_data(int size, float* data, int IDX, int* offset, int* STR, bool negative_str);

std::tuple<float*, float*> zero_tensor(int size, int IDX, int* offset, int* STR, bool negative_str);

float* zero_tensor(int IDX, int* EXT);

float* copy_tensor_data(int IDX, int* EXT, float* data);

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

void run_tblis_mult(int IDXA, int* EXTA, int* STRA, float* A,
                    int IDXB, int* EXTB, int* STRB, float* B,
                    int IDXC, int* EXTC, int* STRC, float* C,
                    int IDXD, int* EXTD, int* STRD, float* D,
                    float ALPHA, float BETA, bool FA, bool FB, bool FC, char* EINSUM) {
    tblis::tblis_tensor tblis_A;
    tblis::len_type len_A[IDXA];
    tblis::stride_type stride_A[IDXA];
    for (int i = 0; i < IDXA; i++)
    {
        len_A[i] = EXTA[i];
        stride_A[i] = STRA[i];
    }

    tblis::tblis_tensor tblis_B;
    tblis::len_type len_B[IDXB];
    tblis::stride_type stride_B[IDXB];
    for (int i = 0; i < IDXB; i++)
    {
        len_B[i] = EXTB[i];
        stride_B[i] = STRB[i];
    }

    tblis::tblis_tensor tblis_C;
    tblis::len_type len_C[IDXC];
    tblis::stride_type stride_C[IDXC];
    for (int i = 0; i < IDXC; i++)
    {
        len_C[i] = EXTC[i];
        stride_C[i] = STRC[i];
    }

    tblis::tblis_tensor tblis_D;
    tblis::len_type len_D[IDXD];
    tblis::stride_type stride_D[IDXD];
    for (int i = 0; i < IDXD; i++)
    {
        len_D[i] = EXTD[i];
        stride_D[i] = STRD[i];
    }

    tblis::tblis_init_tensor_s(&tblis_A, IDXA, len_A, A, stride_A);
    tblis::tblis_init_tensor_s(&tblis_B, IDXB, len_B, B, stride_B);
    tblis::tblis_init_tensor_scaled_s(&tblis_C, BETA, IDXC, len_C, C, stride_C);
    tblis::tblis_init_tensor_scaled_s(&tblis_D, 0, IDXD, len_D, D, stride_D);

    std::string einsum(EINSUM);
    std::string indices = einsum.substr(0, einsum.find("->"));
    std::string indices_A = indices.substr(0, indices.find(","));
    std::string indices_B = indices.substr(indices.find(",")+1, indices.size());
    std::string indices_D = einsum.substr(einsum.find("->")+2, einsum.size());

    indices_A.erase(std::remove_if(indices_A.begin(), indices_A.end(), ::isspace), indices_A.end());
    indices_B.erase(std::remove_if(indices_B.begin(), indices_B.end(), ::isspace), indices_B.end());
    indices_D.erase(std::remove_if(indices_D.begin(), indices_D.end(), ::isspace), indices_D.end());

    tblis::label_type idx_A[indices_A.size() + 1];
    tblis::label_type idx_B[indices_B.size() + 1];
    tblis::label_type idx_D[indices_D.size() + 1];
    idx_A[indices_A.size()] = '\0';
    idx_B[indices_B.size()] = '\0';
    idx_D[indices_D.size()] = '\0';
    for (int i = 0; i < indices_A.size(); i++)
    {
        idx_A[i] = indices_A[i];
    }
    for (int i = 0; i < indices_B.size(); i++)
    {
        idx_B[i] = indices_B[i];
    }
    for (int i = 0; i < indices_D.size(); i++)
    {
        idx_D[i] = indices_D[i];
    }

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, idx_D);

    tblis::tblis_tensor_mult(tblis_single, NULL, &tblis_A, idx_A, &tblis_B, idx_B, &tblis_D, idx_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_C, idx_D);

    tblis::tblis_init_tensor_scaled_s(&tblis_D, ALPHA, IDXD, len_D, D, stride_D);

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, idx_D);
    
    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, idx_D, &tblis_D, idx_D);
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

std::tuple<int, int*, int*, float*,
           int, int*, int*, float*,
           int, int*, int*, float*,
           int, int*, int*, float*,
           float, float, bool, bool, bool, char*,
           float*, float*, float*, float*,
           int, int, int, int,
           int*, int*, int*, int*> generate_contraction(int IDXA = -1, int IDXB = -1,
                                                        int IDXD = randi(0, 5), int contractions = randi(0, 5),
                                                        bool equal_extents = false, bool lower_extents = false,
                                                        bool lower_idx = false, bool negative_str = false) {
    if (IDXA == -1 && IDXB == -1)
    {
        IDXA = randi(0, IDXD);
        IDXB = IDXD - IDXA;
        IDXA = IDXA + contractions;
        IDXB = IDXB + contractions;
    }
    else if (IDXA == -1)
    {
        contractions = contractions > IDXB ? randi(0, IDXB) : contractions;
        IDXD = IDXD < IDXB - contractions ? IDXB - contractions + randi(0, 5) : IDXD;
        IDXA = contractions*2 + IDXD - IDXB;
    }
    else if (IDXB == -1)
    {
        contractions = contractions > IDXA ? randi(0, IDXA) : contractions;
        IDXD = IDXD < IDXA - contractions ? IDXA - contractions + randi(0, 5) : IDXD;
        IDXB = contractions*2 + IDXD - IDXA;
    }
    else
    {
        contractions = contractions > std::min(IDXA, IDXB) ? randi(0, std::min(IDXA, IDXB)) : contractions;
        IDXD = IDXA + IDXB - contractions * 2;
    }
    int IDXC = IDXD;    

    char indicesA[IDXA];
    for (int i = 0; i < IDXA; i++)
    {
        indicesA[i] = 'a' + i;
    }
    if (IDXA > 0) {
        std::shuffle(indicesA, indicesA + IDXA, std::default_random_engine());
    }
    
    char indicesB[IDXB];
    int contracted_indices[contractions];
    for (int i = 0; i < contractions; i++)
    {
        indicesB[i] = indicesA[i];
        contracted_indices[i] = indicesA[i];
    }
    for (int i = 0; i < IDXB - contractions; i++)
    {
        indicesB[i + contractions] = 'a' + IDXA + i;
    }
    if (IDXB > 0) {
        std::shuffle(indicesB, indicesB + IDXB, std::default_random_engine());
    }
    if (IDXA > 0) {
        std::shuffle(indicesA, indicesA + IDXA, std::default_random_engine());
    }

    char indicesD[IDXD];
    int index = 0;
    for (int j = 0; j < IDXA + IDXB - contractions; j++)
    {
        char idx = 'a' + j;
        bool found = false;
        for (int i = 0; i < contractions; i++)
        {
            if (idx == contracted_indices[i])
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            indicesD[index] = idx;
            index++;
        }
    }
    if (IDXD > 0) {
        std::shuffle(indicesD, indicesD + IDXD, std::default_random_engine());
    }

    char* EINSUM = new char[IDXA + IDXB + IDXD + 7];
    for (int i = 0; i < IDXA; i++)
    {
        EINSUM[i] = indicesA[i];
    }
    EINSUM[IDXA] = ',';
    EINSUM[IDXA+1] = ' ';
    for (int i = 0; i < IDXB; i++)
    {
        EINSUM[IDXA+2+i] = indicesB[i];
    }
    EINSUM[IDXA+IDXB+2] = ' ';
    EINSUM[IDXA+IDXB+3] = '-';
    EINSUM[IDXA+IDXB+4] = '>';
    EINSUM[IDXA+IDXB+5] = ' ';
    for (int i = 0; i < IDXD; i++)
    {
        EINSUM[IDXA+IDXB+6+i] = indicesD[i];
    }
    EINSUM[IDXA+IDXB+IDXD+6] = '\0';

    int* EXTA = new int[IDXA];
    int* EXTB = new int[IDXB];
    int* EXTD = new int[IDXD];
    int extent = randi(1, 5);
    for (int i = 0; i < IDXA; i++)
    {
        EXTA[i] = equal_extents ? randi(1, 5) : extent;
    }
    for (int i = 0; i < IDXB; i++)
    {
        int found = -1;
        for (int j = 0; j < IDXA; j++)
        {
            if (indicesB[i] == indicesA[j])
            {
                found = j;
                break;
            }
        }
        if (found != -1)
        {
            EXTB[i] = EXTA[found];
        }
        else
        {
            EXTB[i] = equal_extents ? randi(1, 5) : extent;
        }
    }
    for (int i = 0; i < IDXD; i++)
    {
        int foundA = -1;
        for (int j = 0; j < IDXA; j++)
        {
            if (indicesD[i] == indicesA[j])
            {
                foundA = j;
                break;
            }
        }

        int foundB = -1;
        for (int j = 0; j < IDXB; j++)
        {
            if (indicesD[i] == indicesB[j])
            {
                foundB = j;
                break;
            }
        }

        if (foundA != -1)
        {
            EXTD[i] = EXTA[foundA];
        }
        else if (foundB != -1)
        {
            EXTD[i] = EXTB[foundB];
        }
        else
        {
            std::cout << "Error: Index not found" << std::endl;
        }
    }
    int* EXTC = new int[IDXC];
    std::copy(EXTD, EXTD + IDXD, EXTC);

    int outer_IDXA = lower_idx ? IDXA + randi(1, 5) : IDXA;
    int outer_IDXB = lower_idx ? IDXB + randi(1, 5) : IDXB;
    int outer_IDXC = lower_idx ? IDXC + randi(1, 5) : IDXC;
    int outer_IDXD = lower_idx ? IDXD + randi(1, 5) : IDXD;
    int outer_EXTA[outer_IDXA];
    int outer_EXTB[outer_IDXB];
    int outer_EXTC[outer_IDXC];
    int outer_EXTD[outer_IDXD];
    int* STRA = new int[IDXA];
    int* STRB = new int[IDXB];
    int* STRC = new int[IDXC];
    int* STRD = new int[IDXD];
    int* offsetA = new int[IDXA];
    int* offsetB = new int[IDXB];
    int* offsetC = new int[IDXC];
    int* offsetD = new int[IDXD];
    int sizeA = 1;
    int sizeB = 1;
    int sizeC = 1;
    int sizeD = 1;

    int str = negative_str ? -1 : 1;
    int idx = 0;
    for (int i = 0; i < outer_IDXA; i++)
    {
        if ((randf(0, 1) < (float)IDXA/(float)outer_IDXA || outer_IDXA - i == IDXA - idx) && IDXA - idx > 0)
        {
            int extension = randi(1, 5);
            outer_EXTA[i] = lower_extents ? EXTA[idx] + extension : EXTA[idx];
            offsetA[idx] = lower_extents && extension - EXTA[idx] > 0 ? randi(0, extension - EXTA[idx]) : 0;
            STRA[idx] = str;
            str *= outer_EXTA[i];
            idx++;
        }
        else
        {
            outer_EXTA[i] = lower_extents ? randi(1, 10) : randi(1, 5);
            str *= outer_EXTA[i];
        }
        sizeA *= outer_EXTA[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_IDXB; i++)
    {
        if ((randf(0, 1) < (float)IDXB/(float)outer_IDXB || outer_IDXB - i == IDXB - idx) && IDXB - idx > 0)
        {
            int extension = randi(1, 5);
            outer_EXTB[i] = lower_extents ? EXTB[idx] + extension : EXTB[idx];
            offsetB[idx] = lower_extents && extension - EXTB[idx] > 0 ? randi(0, extension - EXTB[idx]) : 0;
            STRB[idx] = str;
            str *= outer_EXTB[i];
            idx++;
        }
        else
        {
            outer_EXTB[i] = lower_extents ? randi(1, 10) : randi(1, 5);
            str *= outer_EXTB[i];
        }
        sizeB *= outer_EXTB[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_IDXC; i++)
    {
        if ((randf(0, 1) < (float)IDXC/(float)outer_IDXC || outer_IDXC - i == IDXC - idx) && IDXC - idx > 0)
        {
            int extension = randi(1, 5);
            outer_EXTC[i] = lower_extents ? EXTC[idx] + extension : EXTC[idx];
            offsetC[idx] = lower_extents && extension - EXTC[idx] > 0 ? randi(0, extension - EXTC[idx]) : 0;
            STRC[idx] = str;
            str *= outer_EXTC[i];
            idx++;
        }
        else
        {
            outer_EXTC[i] = lower_extents ? randi(1, 10) : randi(1, 5);
            str *= outer_EXTC[i];
        }
        sizeC *= outer_EXTC[i];
    }
    str = negative_str ? -1 : 1;
    idx = 0;
    for (int i = 0; i < outer_IDXD; i++)
    {
        if ((randf(0, 1) < (float)IDXD/(float)outer_IDXD || outer_IDXD - i == IDXD - idx) && IDXD - idx > 0)
        {
            int extension = randi(1, 5);
            outer_EXTD[i] = lower_extents ? EXTD[idx] + extension : EXTD[idx];
            offsetD[idx] = lower_extents && extension - EXTD[idx] > 0 ? randi(0, extension - EXTD[idx]) : 0;
            STRD[idx] = str;
            str *= outer_EXTD[i];
            idx++;
        }
        else
        {
            outer_EXTD[i] = lower_extents ? randi(1, 10) : randi(1, 5);
        }
        sizeD *= outer_EXTD[i];
    }

    float* dataA = new float[sizeA];
    float* dataB = new float[sizeB];
    float* dataC = new float[sizeC];
    float* dataD = new float[sizeD];

    for (int i = 0; i < sizeA; i++)
    {
        dataA[i] = randf(0, 1);
    }
    for (int i = 0; i < sizeB; i++)
    {
        dataB[i] = randf(0, 1);
    }
    for (int i = 0; i < sizeC; i++)
    {
        dataC[i] = randf(0, 1);
    }
    for (int i = 0; i < sizeD; i++)
    {
        dataD[i] = randf(0, 1);
    }

    float* A = negative_str ? dataA + sizeA - 1 : dataA;
    float* B = negative_str ? dataB + sizeB - 1 : dataB;
    float* C = negative_str ? dataC + sizeC - 1 : dataC;
    float* D = negative_str ? dataD + sizeD - 1 : dataD;

    for (int i = 0; i < IDXA; i++)
    {
        A += offsetA[i] * STRA[i];
    }
    for (int i = 0; i < IDXB; i++)
    {
        B += offsetB[i] * STRB[i];
    }
    for (int i = 0; i < IDXC; i++)
    {
        C += offsetC[i] * STRC[i];
    }
    for (int i = 0; i < IDXD; i++)
    {
        D += offsetD[i] * STRD[i];
    }

    float ALPHA = randf(0, 1);
    float BETA = randf(0, 1);

    bool FA = false;
    bool FB = false;
    bool FC = false;

    return {IDXA, EXTA, STRA, A,
            IDXB, EXTB, STRB, B,
            IDXC, EXTC, STRC, C,
            IDXD, EXTD, STRD, D,
            ALPHA, BETA, FA, FB, FC, EINSUM,
            dataA, dataB, dataC, dataD,
            sizeA, sizeB, sizeC, sizeD,
            offsetA, offsetB, offsetC, offsetD};
}

std::tuple<float*, float*> copy_tensor_data(int size, float* data, int IDX, int* offset, int* STR, bool negative_str = false) {
    float* dataA = new float[size];
    std::copy(data, data + size, dataA);
    float* A = negative_str ? dataA + size - 1 : dataA;
    for (int i = 0; i < IDX; i++)
    {
        A += offset[i] * STR[i];
    }
    return {A, dataA};
}

std::tuple<float*, float*> zero_tensor(int size, int IDX, int* offset, int* STR, bool negative_str = false) {
    float* dataA = new float[size];
    for (int i = 0; i < size; i++)
    {
        dataA[i] = 0;
    }
    float* A = negative_str ? dataA + size - 1 : dataA;
    for (int i = 0; i < IDX; i++)
    {
        A += offset[i] * STR[i];
    }
    return {A, dataA};
}

float* copy_tensor_data(int IDX, int* EXT, float* data) {
    int size = 1;
    for (int i = 0; i < IDX; i++)
    {
        size *= EXT[i];
    }
    float* copy = new float[size];
    for (int i = 0; i < size; i++)
    {
        copy[i] = data[i];
    }
    return copy;
}

float* zero_tensor(int IDX, int* EXT) {
    int size = 1;
    for (int i = 0; i < IDX; i++)
    {
        size *= EXT[i];
    }
    float* zero = new float[size];
    for (int i = 0; i < size; i++)
    {
        zero[i] = 0;
    }
    return zero;
}

int calculate_tensor_size(int IDX, int* EXT) {
    int size = 1;
    for (int i = 0; i < IDX; i++)
    {
        size *= EXT[i];
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

void rotate_output_indices(char* EINSUM, int IDXA, int IDXB, int IDXC, int* EXTC, int* STRC, int IDXD, int* EXTD, int* STRD) {
    char index = EINSUM[IDXA + IDXB + 6];
    int extent = EXTD[0];
    STRD[0] = 1;
    STRC[0] = 1;
    for (int i = 0; i < IDXD - 1; i++)
    {
        EINSUM[IDXA + IDXB + 6 + i] = EINSUM[IDXA + IDXB + 7 + i];
        EXTD[i] = EXTD[i+1];
        STRD[i + 1] = STRD[i] * EXTD[i];
        EXTC[i] = EXTC[i+1];
        STRC[i + 1] = STRC[i] * EXTC[i];
    }
    EINSUM[IDXA + IDXB + 6 + IDXD - 1] = index;
    EXTD[IDXD-1] = extent;
    EXTC[IDXD-1] = extent;
}

void print_tensor(int IDX, int* EXT, int* STR, float* data) {
    std::cout << "IDX: " << IDX << std::endl;
    std::cout << "EXT: ";
    for (int i = 0; i < IDX; i++)
    {
        std::cout << EXT[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "STR: ";
    for (int i = 0; i < IDX; i++)
    {
        std::cout << STR[i] << " ";
    }
    std::cout << std::endl;
    int coord[IDX];
    for (int i = 0; i < IDX; i++)
    {
        coord[i] = 0;
    }
    int size = 1;
    for (int i = 0; i < IDX; i++)
    {
        size *= EXT[i];
    }
    for (int i = 0; i < size; i++)
    {
        std::cout << data[i] << " ";
        coord[0]++;
        for (int j = 0; j < IDX - 1; j++)
        {
            if (coord[j] == EXT[j])
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
    int IDX = randi(0, 5);
    int* EXT = new int[IDX];
    int* STR = new int[IDX];
    int size = 1;
    for (int i = 0; i < IDX; i++)
    {
        EXT[i] = randi(1, 5);
        size *= EXT[i];
    }
    if (IDX > 0) {
        STR[0] = 1;
    }
    for (int i = 1; i < IDX; i++)
    {
        STR[i] = STR[i-1] * EXT[i-1];
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

    float ALPHA = randf(0, 1);
    float BETA = randf(0, 1);

    char* EINSUM = new char[IDX*3 + 7];
    for (int i = 0; i < IDX; i++)
    {
        EINSUM[i] = 'a' + i;
        EINSUM[IDX+2+i] = 'a' + i;
        EINSUM[IDX*2+6+i] = 'a' + i;
    }
    EINSUM[IDX] = ',';
    EINSUM[IDX+1] = ' ';
    EINSUM[IDX*2+2] = ' ';
    EINSUM[IDX*2+3] = '-';
    EINSUM[IDX*2+4] = '>';
    EINSUM[IDX*2+5] = ' ';
    EINSUM[IDX*3+6] = '\0';

    float* E = copy_tensor_data(IDX, EXT, D);

    PRODUCT(IDX, EXT, STR, A, IDX, EXT, STR, B, IDX, EXT, STR, C, IDX, EXT, STR, D, ALPHA, BETA, true, true, true, EINSUM);

    run_tblis_mult(IDX, EXT, STR, A, IDX, EXT, STR, B, IDX, EXT, STR, C, IDX, EXT, STR, E, ALPHA, BETA, true, true, true, EINSUM);

    bool result = compare_tensors(D, E, size);

    delete[] EXT;
    delete[] STR;
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] D;
    delete[] E;
    delete[] EINSUM;

    return result;
}

bool test_contraction() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction();

    auto [E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] dataE;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;

    return result;
}

bool test_commutativity() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction();

    auto [E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    auto [F, dataF] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    auto[G, dataG] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    auto[C1, dataC1] = copy_tensor_data(sizeC, dataC, IDXC, offsetC, STRC);

    auto[C2, dataC2] = copy_tensor_data(sizeC, dataC, IDXC, offsetC, STRC);

    auto[C3, dataC3] = copy_tensor_data(sizeC, dataC, IDXC, offsetC, STRC);

    auto[C4, dataC4] = copy_tensor_data(sizeC, dataC, IDXC, offsetC, STRC);

    char* swapped_EINSUM = swap_indices(EINSUM, IDXA, IDXB, IDXD);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C1, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C2, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    PRODUCT(IDXB, EXTB, STRB, B, IDXA, EXTA, STRA, A, IDXC, EXTC, STRC, C3, IDXD, EXTD, STRD, F, ALPHA, BETA, FB, FA, FC, swapped_EINSUM);

    run_tblis_mult(IDXB, EXTB, STRB, B, IDXA, EXTA, STRA, A, IDXC, EXTC, STRC, C4, IDXD, EXTD, STRD, G, ALPHA, BETA, FA, FB, FC, swapped_EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD) && compare_tensors(dataF, dataG, sizeD) && compare_tensors(dataD, dataF, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;
    delete[] dataF;
    delete[] dataG;
    delete[] dataC1;
    delete[] dataC2;
    delete[] dataC3;
    delete[] dataC4;
    delete[] swapped_EINSUM;


    return result;
}

bool test_permutations() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction();
    
    bool result = true;

    for (int i = 0; i < IDXD; i++)
    {
        auto[C1, dataC1] = copy_tensor_data(sizeC, dataC, IDXC, offsetC, STRC);
        PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C1, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);
        auto[C2, dataC2] = copy_tensor_data(sizeC, dataC, IDXC, offsetC, STRC);
        auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);
        run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C2, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);
        result = result && compare_tensors(dataD, dataE, sizeD);
        rotate_output_indices(EINSUM, IDXA, IDXB, IDXC, EXTC, STRC, IDXD, EXTD, STRD);
        delete[] dataC1;
        delete[] dataC2;
        delete[] dataE;
    }
    
    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;

    return result;
}

bool test_equal_extents() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(true);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_outer_product() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(-1, -1, randi(0, 5), 0);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);
    
    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_full_contraction() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(-1, -1, 0);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_zero_dim_tensor_contraction() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(0);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_one_dim_tensor_contraction() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(1);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_subtensor_same_idx() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(-1, -1, randi(0, 5), randi(0, 5), false, true);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_subtensor_lower_idx() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(-1, -1, randi(0, 5), randi(0, 5), false, true, true);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_negative_strides() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(-1, -1, randi(0, 5), randi(0, 5), false, false, false, true);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD, true);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_negative_strides_subtensor_same_idx() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(-1, -1, randi(0, 5), randi(0, 5), false, true, false, true);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD, true);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}

bool test_negative_strides_subtensor_lower_idx() {
    auto [IDXA, EXTA, STRA, A,
          IDXB, EXTB, STRB, B,
          IDXC, EXTC, STRC, C,
          IDXD, EXTD, STRD, D,
          ALPHA, BETA, FA, FB, FC, EINSUM,
          dataA, dataB, dataC, dataD,
          sizeA, sizeB, sizeC, sizeD,
          offsetA, offsetB, offsetC, offsetD] = generate_contraction(-1, -1, randi(0, 5), randi(0, 5), false, true, true, true);
    
    auto[E, dataE] = copy_tensor_data(sizeD, dataD, IDXD, offsetD, STRD, true);

    PRODUCT(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, D, ALPHA, BETA, FA, FB, FC, EINSUM);

    run_tblis_mult(IDXA, EXTA, STRA, A, IDXB, EXTB, STRB, B, IDXC, EXTC, STRC, C, IDXD, EXTD, STRD, E, ALPHA, BETA, FA, FB, FC, EINSUM);

    bool result = compare_tensors(dataD, dataE, sizeD);

    delete[] EXTA;
    delete[] EXTB;
    delete[] EXTC;
    delete[] EXTD;
    delete[] STRA;
    delete[] STRB;
    delete[] STRC;
    delete[] STRD;
    delete[] EINSUM;
    delete[] dataA;
    delete[] dataB;
    delete[] dataC;
    delete[] dataD;
    delete[] offsetA;
    delete[] offsetB;
    delete[] offsetC;
    delete[] offsetD;
    delete[] dataE;

    return result;
}