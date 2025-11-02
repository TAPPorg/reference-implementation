/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#include "test.h"

int main(int argc, char const *argv[])
{
    srand(time(NULL)); 
    std::cout << "Contraction Complex Double Precision: " << str(test_contraction_complex_double_precision()) << std::endl;
     //std::cout << "Contraction Double Precision: " << str(test_contraction_double_precision()) << std::endl;
    TAPP_finalize();
    exit(0);
    std::cout << "Hadamard Product: " << str(test_hadamard_product()) << std::endl;
    std::cout << "Contraction: " << str(test_contraction()) << std::endl;
    std::cout << "Commutativity: " << str(test_commutativity()) << std::endl;
    std::cout << "Permutations: " << str(test_permutations()) << std::endl;
    std::cout << "Equal Extents: " << str(test_equal_extents()) << std::endl;
    std::cout << "Outer Product: " << str(test_outer_product()) << std::endl;
    std::cout << "Full Contraction: " << str(test_full_contraction()) << std::endl;
    //for(int i=0;i<0;i++)
    std::cout << "Zero Dim Tensor Contraction: " << str(test_zero_dim_tensor_contraction()) << std::endl;
    std::cout << "One Dim Tensor Contraction: " << str(test_one_dim_tensor_contraction()) << std::endl;
    std::cout << "Subtensor Same Index: " << str(test_subtensor_same_idx()) << std::endl;
    std::cout << "Subtensor Lower Index: " << str(test_subtensor_lower_idx()) << std::endl;
    std::cout << "Negative Strides: " << str(test_negative_strides()) << std::endl;
    std::cout << "Negative Strides Subtensor Same Index: " << str(test_negative_strides_subtensor_same_idx()) << std::endl;
    std::cout << "Negative Strides Subtensor Lower Index: " << str(test_negative_strides_subtensor_lower_idx()) << std::endl;
    std::cout << "Mixed Strides: " << str(test_mixed_strides()) << std::endl;
    std::cout << "Mixed Strides Subtensor Same Index: " << str(test_mixed_strides_subtensor_same_idx()) << std::endl;
    std::cout << "Mixed Strides Subtensor Lower Index: " << str(test_mixed_strides_subtensor_lower_idx()) << std::endl;
    std::cout << "Contraction Double Precision: " << str(test_contraction_double_precision()) << std::endl;
    std::cout << "Contraction Complex: " << str(test_contraction_complex()) << std::endl;
    //for(int i=0;i<1;i++)
    std::cout << "Contraction Complex Double Precision: " << str(test_contraction_complex_double_precision()) << std::endl;
    std::cout << "Zero stride: " << str(test_zero_stride()) << std::endl;
    std::cout << "Unique Index: " << str(test_unique_idx()) << std::endl;
    std::cout << "Repeated Index: " << str(test_repeated_idx()) << std::endl;
    std::cout << "Hadamard And Free: " << str(test_hadamard_and_free()) << std::endl;
    std::cout << "Hadamard And Contraction: " << str(test_hadamard_and_contraction()) << std::endl;
    std::cout << "Error: Non Matching Extents: " << str(test_error_non_matching_ext()) << std::endl;
    std::cout << "Error: C Other Structure: " << str(test_error_C_other_structure()) << std::endl;
    std::cout << "Error: Aliasing Within D: " << str(test_error_aliasing_within_D()) << std::endl;
    return 0;
}

void run_tblis_mult_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, float* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, float* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, float* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, float* D, int op_D, int64_t* idx_D,
                    float alpha, float beta)
{
    tblis::len_type* tblis_len_A = translate_extents_to_tblis(nmode_A, extents_A);
    tblis::stride_type* tblis_stride_A = translate_strides_to_tblis(nmode_A, strides_A);
    tblis::tblis_tensor tblis_A;
    tblis::tblis_init_tensor_scaled_s(&tblis_A, alpha, nmode_A, tblis_len_A, A, tblis_stride_A);
    tblis::label_type* tblis_idx_A = translate_idx_to_tblis(nmode_A, idx_A);

    tblis::len_type* tblis_len_B = translate_extents_to_tblis(nmode_B, extents_B);
    tblis::stride_type* tblis_stride_B = translate_strides_to_tblis(nmode_B, strides_B);
    tblis::tblis_tensor tblis_B;
    tblis::tblis_init_tensor_s(&tblis_B, nmode_B, tblis_len_B, B, tblis_stride_B);
    tblis::label_type* tblis_idx_B = translate_idx_to_tblis(nmode_B, idx_B);

    tblis::len_type* tblis_len_C = translate_extents_to_tblis(nmode_C, extents_C);
    tblis::stride_type* tblis_stride_C = translate_strides_to_tblis(nmode_C, strides_C);
    tblis::tblis_tensor tblis_C;
    tblis::tblis_init_tensor_scaled_s(&tblis_C, beta, nmode_C, tblis_len_C, C, tblis_stride_C);
    tblis::label_type* tblis_idx_C = translate_idx_to_tblis(nmode_C, idx_C);
    
    tblis::len_type* tblis_len_D = translate_extents_to_tblis(nmode_D, extents_D);
    tblis::stride_type* tblis_stride_D = translate_strides_to_tblis(nmode_D, strides_D);
    tblis::tblis_tensor tblis_D;
    tblis::tblis_init_tensor_scaled_s(&tblis_D, 0, nmode_D, tblis_len_D, D, tblis_stride_D);
    tblis::label_type* tblis_idx_D = translate_idx_to_tblis(nmode_D, idx_D);

    auto [tblis_A_reduced, tblis_idx_A_reduced, tblis_len_A_reduced, tblis_stride_A_reduced, tblis_data_A_reduced] = contract_unique_idx_s(&tblis_A, tblis_idx_A, nmode_B, tblis_idx_B, nmode_D, tblis_idx_D);
    
    auto [tblis_B_reduced, tblis_idx_B_reduced, tblis_len_B_reduced, tblis_stride_B_reduced, tblis_data_B_reduced] = contract_unique_idx_s(&tblis_B, tblis_idx_B, nmode_A, tblis_idx_A, nmode_D, tblis_idx_D);

    tblis::tblis_tensor_mult(tblis_single, NULL, tblis_A_reduced, tblis_idx_A_reduced, tblis_B_reduced, tblis_idx_B_reduced, &tblis_D, tblis_idx_D);

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, tblis_idx_C, &tblis_D, tblis_idx_D);

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

std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, float*> contract_unique_idx_s(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2)
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

    float* data_reduced = new float[size_reduced];
    for (size_t i = 0; i < size_reduced; i++)
    {
        data_reduced[i] = 0;
    }
    tblis::tblis_init_tensor_s(tblis_reduced, nmode_reduced, len_reduced, data_reduced, stride_reduced);
    tblis::tblis_tensor_add(tblis_single, NULL, tensor, idx, tblis_reduced, idx_reduced);
    return {tblis_reduced, idx_reduced, len_reduced, stride_reduced, data_reduced};
}

void run_tblis_mult_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, double* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, double* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, double* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, double* D, int op_D, int64_t* idx_D,
                    double alpha, double beta)
{
    tblis::len_type* tblis_len_A = translate_extents_to_tblis(nmode_A, extents_A);
    tblis::stride_type* tblis_stride_A = translate_strides_to_tblis(nmode_A, strides_A);
    tblis::tblis_tensor tblis_A;
    tblis::tblis_init_tensor_scaled_d(&tblis_A, alpha, nmode_A, tblis_len_A, A, tblis_stride_A);
    tblis::label_type* tblis_idx_A = translate_idx_to_tblis(nmode_A, idx_A);

    tblis::len_type* tblis_len_B = translate_extents_to_tblis(nmode_B, extents_B);
    tblis::stride_type* tblis_stride_B = translate_strides_to_tblis(nmode_B, strides_B);
    tblis::tblis_tensor tblis_B;
    tblis::tblis_init_tensor_d(&tblis_B, nmode_B, tblis_len_B, B, tblis_stride_B);
    tblis::label_type* tblis_idx_B = translate_idx_to_tblis(nmode_B, idx_B);

    tblis::len_type* tblis_len_C = translate_extents_to_tblis(nmode_C, extents_C);
    tblis::stride_type* tblis_stride_C = translate_strides_to_tblis(nmode_C, strides_C);
    tblis::tblis_tensor tblis_C;
    tblis::tblis_init_tensor_scaled_d(&tblis_C, beta, nmode_C, tblis_len_C, C, tblis_stride_C);
    tblis::label_type* tblis_idx_C = translate_idx_to_tblis(nmode_C, idx_C);
    
    tblis::len_type* tblis_len_D = translate_extents_to_tblis(nmode_D, extents_D);
    tblis::stride_type* tblis_stride_D = translate_strides_to_tblis(nmode_D, strides_D);
    tblis::tblis_tensor tblis_D;
    tblis::tblis_init_tensor_scaled_d(&tblis_D, 0, nmode_D, tblis_len_D, D, tblis_stride_D);
    tblis::label_type* tblis_idx_D = translate_idx_to_tblis(nmode_D, idx_D);

    auto [tblis_A_reduced, tblis_idx_A_reduced, tblis_len_A_reduced, tblis_stride_A_reduced, tblis_data_A_reduced] = contract_unique_idx_d(&tblis_A, tblis_idx_A, nmode_B, tblis_idx_B, nmode_D, tblis_idx_D);
    
    auto [tblis_B_reduced, tblis_idx_B_reduced, tblis_len_B_reduced, tblis_stride_B_reduced, tblis_data_B_reduced] = contract_unique_idx_d(&tblis_B, tblis_idx_B, nmode_A, tblis_idx_A, nmode_D, tblis_idx_D);

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, tblis_idx_C, &tblis_D, tblis_idx_D);
    tblis::tblis_tensor_mult(tblis_single, NULL, tblis_A_reduced, tblis_idx_A_reduced, tblis_B_reduced, tblis_idx_B_reduced, &tblis_D, tblis_idx_D);


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

std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, double*> contract_unique_idx_d(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2)
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
            stride_reduced[nmode_reduced] = nmode_reduced == 0 ? 1 : stride_reduced[nmode_reduced - 1] * tensor->len[nmode_reduced - 1];
            idx_reduced[nmode_reduced] = idx[i];
            size_reduced *= len_reduced[nmode_reduced];
            nmode_reduced++;
        }
    }
    idx_reduced[nmode_reduced] = '\0';

    double* data_reduced = new double[size_reduced];
    for (size_t i = 0; i < size_reduced; i++)
    {
        data_reduced[i] = 0;
    }

    tblis::tblis_init_tensor_d(tblis_reduced, nmode_reduced, len_reduced, data_reduced, stride_reduced);
    tblis::tblis_tensor_add(tblis_single, NULL, tensor, idx, tblis_reduced, idx_reduced);
    return {tblis_reduced, idx_reduced, len_reduced, stride_reduced, data_reduced};
}

void run_tblis_mult_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<float>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<float>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<float>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<float>* D, int op_D, int64_t* idx_D,
                    std::complex<float> alpha, std::complex<float> beta)
{
    tblis::len_type* tblis_len_A = translate_extents_to_tblis(nmode_A, extents_A);
    tblis::stride_type* tblis_stride_A = translate_strides_to_tblis(nmode_A, strides_A);
    tblis::tblis_tensor tblis_A;
    tblis::tblis_init_tensor_scaled_c(&tblis_A, alpha, nmode_A, tblis_len_A, A, tblis_stride_A);
    tblis::label_type* tblis_idx_A = translate_idx_to_tblis(nmode_A, idx_A);
    tblis_A.conj = op_A;

    tblis::len_type* tblis_len_B = translate_extents_to_tblis(nmode_B, extents_B);
    tblis::stride_type* tblis_stride_B = translate_strides_to_tblis(nmode_B, strides_B);
    tblis::tblis_tensor tblis_B;
    tblis::tblis_init_tensor_c(&tblis_B, nmode_B, tblis_len_B, B, tblis_stride_B);
    tblis::label_type* tblis_idx_B = translate_idx_to_tblis(nmode_B, idx_B);
    tblis_B.conj = op_B;

    tblis::len_type* tblis_len_C = translate_extents_to_tblis(nmode_C, extents_C);
    tblis::stride_type* tblis_stride_C = translate_strides_to_tblis(nmode_C, strides_C);
    tblis::tblis_tensor tblis_C;
    tblis::tblis_init_tensor_scaled_c(&tblis_C, beta, nmode_C, tblis_len_C, C, tblis_stride_C);
    tblis::label_type* tblis_idx_C = translate_idx_to_tblis(nmode_C, idx_C);
    
    tblis::len_type* tblis_len_D = translate_extents_to_tblis(nmode_D, extents_D);
    tblis::stride_type* tblis_stride_D = translate_strides_to_tblis(nmode_D, strides_D);
    tblis::tblis_tensor tblis_D;
    tblis::tblis_init_tensor_scaled_c(&tblis_D, 0, nmode_D, tblis_len_D, D, tblis_stride_D);
    tblis::label_type* tblis_idx_D = translate_idx_to_tblis(nmode_D, idx_D);

    auto [tblis_A_reduced, tblis_idx_A_reduced, tblis_len_A_reduced, tblis_stride_A_reduced, tblis_data_A_reduced] = contract_unique_idx_c(&tblis_A, tblis_idx_A, nmode_B, tblis_idx_B, nmode_D, tblis_idx_D);
    
    auto [tblis_B_reduced, tblis_idx_B_reduced, tblis_len_B_reduced, tblis_stride_B_reduced, tblis_data_B_reduced] = contract_unique_idx_c(&tblis_B, tblis_idx_B, nmode_A, tblis_idx_A, nmode_D, tblis_idx_D);

    tblis_C.conj = op_C;

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, tblis_idx_C, &tblis_D, tblis_idx_D);
    tblis::tblis_tensor_mult(tblis_single, NULL, tblis_A_reduced, tblis_idx_A_reduced, tblis_B_reduced, tblis_idx_B_reduced, &tblis_D, tblis_idx_D);


    tblis_D.conj = op_D;

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, tblis_idx_D);

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

std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, std::complex<float>*> contract_unique_idx_c(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2)
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
            stride_reduced[nmode_reduced] = nmode_reduced == 0 ? 1 : stride_reduced[nmode_reduced - 1] * tensor->len[nmode_reduced - 1];
            idx_reduced[nmode_reduced] = idx[i];
            size_reduced *= len_reduced[nmode_reduced];
            nmode_reduced++;
        }
    }
    idx_reduced[nmode_reduced] = '\0';

    std::complex<float>* data_reduced = new std::complex<float>[size_reduced];
    for (size_t i = 0; i < size_reduced; i++)
    {
        data_reduced[i] = 0;
    }

    tblis::tblis_init_tensor_c(tblis_reduced, nmode_reduced, len_reduced, data_reduced, stride_reduced);
    tblis::tblis_tensor_add(tblis_single, NULL, tensor, idx, tblis_reduced, idx_reduced);
    return {tblis_reduced, idx_reduced, len_reduced, stride_reduced, data_reduced};
}

void run_tblis_mult_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<double>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<double>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<double>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<double>* D, int op_D, int64_t* idx_D,
                    std::complex<double> alpha, std::complex<double> beta)
{
    tblis::len_type* tblis_len_A = translate_extents_to_tblis(nmode_A, extents_A);
    tblis::stride_type* tblis_stride_A = translate_strides_to_tblis(nmode_A, strides_A);
    tblis::tblis_tensor tblis_A;
    tblis::tblis_init_tensor_scaled_z(&tblis_A, alpha, nmode_A, tblis_len_A, A, tblis_stride_A);
    tblis::label_type* tblis_idx_A = translate_idx_to_tblis(nmode_A, idx_A);
    tblis_A.conj = op_A;

    tblis::len_type* tblis_len_B = translate_extents_to_tblis(nmode_B, extents_B);
    tblis::stride_type* tblis_stride_B = translate_strides_to_tblis(nmode_B, strides_B);
    tblis::tblis_tensor tblis_B;
    tblis::tblis_init_tensor_z(&tblis_B, nmode_B, tblis_len_B, B, tblis_stride_B);
    tblis::label_type* tblis_idx_B = translate_idx_to_tblis(nmode_B, idx_B);
    tblis_B.conj = op_B;

    tblis::len_type* tblis_len_C = translate_extents_to_tblis(nmode_C, extents_C);
    tblis::stride_type* tblis_stride_C = translate_strides_to_tblis(nmode_C, strides_C);
    tblis::tblis_tensor tblis_C;
    tblis::tblis_init_tensor_scaled_z(&tblis_C, beta, nmode_C, tblis_len_C, C, tblis_stride_C);
    tblis::label_type* tblis_idx_C = translate_idx_to_tblis(nmode_C, idx_C);
    
    tblis::len_type* tblis_len_D = translate_extents_to_tblis(nmode_D, extents_D);
    tblis::stride_type* tblis_stride_D = translate_strides_to_tblis(nmode_D, strides_D);
    tblis::tblis_tensor tblis_D;
    tblis::tblis_init_tensor_scaled_z(&tblis_D, 0, nmode_D, tblis_len_D, D, tblis_stride_D);
    tblis::label_type* tblis_idx_D = translate_idx_to_tblis(nmode_D, idx_D);

    auto [tblis_A_reduced, tblis_idx_A_reduced, tblis_len_A_reduced, tblis_stride_A_reduced, tblis_data_A_reduced] = contract_unique_idx_z(&tblis_A, tblis_idx_A, nmode_B, tblis_idx_B, nmode_D, tblis_idx_D);
    
    auto [tblis_B_reduced, tblis_idx_B_reduced, tblis_len_B_reduced, tblis_stride_B_reduced, tblis_data_B_reduced] = contract_unique_idx_z(&tblis_B, tblis_idx_B, nmode_A, tblis_idx_A, nmode_D, tblis_idx_D);

    tblis_C.conj = op_C;

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, tblis_idx_C, &tblis_D, tblis_idx_D);

    tblis::tblis_tensor_mult(tblis_single, NULL, tblis_A_reduced, tblis_idx_A_reduced, tblis_B_reduced, tblis_idx_B_reduced, &tblis_D, tblis_idx_D);


    tblis_D.conj = op_D;

    tblis::tblis_tensor_scale(tblis_single, NULL, &tblis_D, tblis_idx_D);

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

std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, std::complex<double>*> contract_unique_idx_z(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2)
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
            stride_reduced[nmode_reduced] = nmode_reduced == 0 ? 1 : stride_reduced[nmode_reduced - 1] * tensor->len[nmode_reduced - 1];
            idx_reduced[nmode_reduced] = idx[i];
            size_reduced *= len_reduced[nmode_reduced];
            nmode_reduced++;
        }
    }
    idx_reduced[nmode_reduced] = '\0';

    std::complex<double>* data_reduced = new std::complex<double>[size_reduced];
    for (size_t i = 0; i < size_reduced; i++)
    {
        data_reduced[i] = 0;
    }

    tblis::tblis_init_tensor_z(tblis_reduced, nmode_reduced, len_reduced, data_reduced, stride_reduced);
    tblis::tblis_tensor_add(tblis_single, NULL, tensor, idx, tblis_reduced, idx_reduced);
    return {tblis_reduced, idx_reduced, len_reduced, stride_reduced, data_reduced};
}

tblis::len_type* translate_extents_to_tblis(int nmode, int64_t* extents)
{
    tblis::len_type* tblis_len = new tblis::len_type[nmode];
    for (int i = 0; i < nmode; i++)
    {
        tblis_len[i] = extents[i];
    }
    return tblis_len;
}

tblis::stride_type* translate_strides_to_tblis(int nmode, int64_t* strides)
{
    tblis::stride_type* tblis_stride = new tblis::stride_type[nmode];
    for (int i = 0; i < nmode; i++)
    {
        tblis_stride[i] = strides[i];
    }
    return tblis_stride;
}

tblis::label_type* translate_idx_to_tblis(int nmode, int64_t* idx)
{
    tblis::label_type* tblis_idx = new tblis::label_type[nmode + 1];
    for (int i = 0; i < nmode; i++)
    {
        tblis_idx[i] = idx[i];
    }
    tblis_idx[nmode] = '\0';
    return tblis_idx;
}

bool compare_tensors_s(float* A, float* B, int size)
{
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

bool compare_tensors_d(double* A, double* B, int size)
{
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

bool compare_tensors_c(std::complex<float>* A, std::complex<float>* B, int size)
{
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

bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size)
{
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        double rel_diff_r = abs((A[i].real() - B[i].real()) / (A[i].real() > B[i].real() ? A[i].real() : B[i].real()));
        double rel_diff_i = abs((A[i].imag() - B[i].imag()) / (A[i].imag() > B[i].imag() ? A[i].imag() : B[i].imag()));
        if (rel_diff_r > 0.00005 || rel_diff_i > 0.00005) //0.00005 // 0.0000000005
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
           int64_t, int64_t, int64_t, int64_t> generate_contraction_s(int nmode_A = -1, int nmode_B = -1,
                                                        int nmode_D = randi(0, 4), int contractions = randi(0, 4),
                                                        int min_extent = 1, bool equal_extents = false,
                                                        bool lower_extents = false, bool lower_nmode = false,
                                                        bool negative_str = false, bool unique_idx = false,
                                                        bool repeated_idx = false, bool mixed_str = false)
{
    if (repeated_idx && nmode_D < 2)
    {
        nmode_D = randi(2, 4);
    }
    if (nmode_A == -1 && nmode_B == -1)
    {
        nmode_A = repeated_idx ? randi(1, nmode_D - 1) : randi(0, nmode_D);
        nmode_B = nmode_D - nmode_A;
        nmode_A = nmode_A + contractions;
        nmode_B = nmode_B + contractions;
    }
    else if (nmode_A == -1)
    {
        contractions = contractions > nmode_B ? (repeated_idx ? randi(0, nmode_B - 1) : randi(0, nmode_B)) : contractions;
        nmode_D = nmode_D < nmode_B - contractions ? nmode_B - contractions + (repeated_idx ? randi(1, 4) : randi(0, 4)) : nmode_D;
        nmode_A = contractions*2 + nmode_D - nmode_B;
    }
    else if (nmode_B == -1)
    {
        contractions = contractions > nmode_A ? (repeated_idx ? randi(0, nmode_A - 1) : randi(0, nmode_A)) : contractions;
        nmode_D = nmode_D < nmode_A - contractions ? nmode_A - contractions + (repeated_idx ? randi(1, 4) : randi(0, 4)) : nmode_D;
        nmode_B = contractions*2 + nmode_D - nmode_A;
    }
    else
    {
        contractions = contractions > std::min(nmode_A, nmode_B) ? randi(0, std::min(nmode_A, nmode_B)) : contractions;
        nmode_D = nmode_A + nmode_B - contractions * 2;
    }

    int unique_idx_A = unique_idx ? randi(1, 3) : 0;

    int unique_idx_B = unique_idx ? randi(1, 3) : 0;

    nmode_A += unique_idx_A;
    nmode_B += unique_idx_B;

    int repeated_idx_A = repeated_idx ? randi(1, 4) : 0;
    int repeated_idx_B = repeated_idx ? randi(1, 4) : 0;
    int repeated_idx_D = repeated_idx ? randi(1, 4) : 0;

    nmode_A += repeated_idx_A;
    nmode_B += repeated_idx_B;
    nmode_D += repeated_idx_D;
    
    int nmode_C = nmode_D;

    int64_t* idx_A = new int64_t[nmode_A];
    for (int i = 0; i < nmode_A - repeated_idx_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A - repeated_idx_A, std::default_random_engine());
    }

    
    int64_t* idx_B = new int64_t[nmode_B];
    int idx_contracted[contractions];
    for (int i = 0; i < contractions; i++)
    {
        idx_B[i] = idx_A[i];
        idx_contracted[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B - contractions - repeated_idx_B; i++)
    {
        idx_B[i + contractions] = 'a' + nmode_A - repeated_idx_A + i;
    }

    if (nmode_B > 0)
    {
        std::shuffle(idx_B, idx_B + nmode_B - repeated_idx_B, std::default_random_engine());
    }
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A - repeated_idx_A, std::default_random_engine());
    }

    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    int index = 0;
    int index_origin = 0;
    for (int i = 0; i < nmode_A - repeated_idx_A - unique_idx_A - contractions; i++)
    {
        for (int j = index_origin; j < nmode_A - repeated_idx_A; j++)
        {
            bool is_contracted = false;
            for (int k = 0; k < contractions; k++)
            {
                if (idx_A[j] == idx_contracted[k])
                {
                    is_contracted = true;
                    break;
                }
            }
            if (!is_contracted)
            {
                index_origin = j;
                break;
            }
        }
        idx_D[index] = idx_A[index_origin];
        index_origin++;
        index++;
    }
    index_origin = 0;
    for (int i = 0; i < nmode_B - repeated_idx_B - unique_idx_B - contractions; i++)
    {
        for (int j = index_origin; j < nmode_B - repeated_idx_B; j++)
        {
            bool is_contracted = false;
            for (int k = 0; k < contractions; k++)
            {
                if (idx_B[j] == idx_contracted[k])
                {
                    is_contracted = true;
                    break;
                }
            }
            if (!is_contracted)
            {
                index_origin = j;
                break;
            }
        }
        idx_D[index] = idx_B[index_origin];
        index_origin++;
        index++;
    }
    
    //Add repeated idx
    for (int i = 0; i < repeated_idx_A; i++)
    {
        idx_A[i + nmode_A - repeated_idx_A] = idx_A[randi(0, nmode_A - repeated_idx_A - 1)];
    }
    for (int i = 0; i < repeated_idx_B; i++)
    {
        idx_B[i + nmode_B - repeated_idx_B] = idx_B[randi(0, nmode_B - repeated_idx_B - 1)];
    }
    for (int i = 0; i < repeated_idx_D; i++)
    {
        idx_D[i + nmode_D - repeated_idx_D] = idx_D[randi(0, nmode_D - repeated_idx_D - 1)];
    }
    
    //Randomize order of idx
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }
    if (nmode_B > 0)
    {
        std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    }
    if (nmode_D > 0)
    {
        std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    }
    std::copy(idx_D, idx_D + nmode_D, idx_C);

    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    int64_t extent = randi(min_extent, 4);
    time_t time_seed = time(NULL);
    for (int i = 0; i < nmode_A; i++)
    {
        srand(time_seed * idx_A[i]);
        extents_A[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    for (int i = 0; i < nmode_B; i++)
    {
        srand(time_seed * idx_B[i]);
        extents_B[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    for (int i = 0; i < nmode_D; i++)
    {
        srand(time_seed * idx_D[i]);
        extents_D[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);

    int outer_nmode_A = lower_nmode ? nmode_A + randi(1, 4) : nmode_A;
    int outer_nmode_B = lower_nmode ? nmode_B + randi(1, 4) : nmode_B;
    int outer_nmode_C = lower_nmode ? nmode_C + randi(1, 4) : nmode_C;
    int outer_nmode_D = lower_nmode ? nmode_D + randi(1, 4) : nmode_D;

    int* stride_signs_A = choose_stride_signs(nmode_A, negative_str, mixed_str);
    int* stride_signs_B = choose_stride_signs(nmode_B, negative_str, mixed_str);
    int* stride_signs_C = choose_stride_signs(nmode_C, negative_str, mixed_str);
    int* stride_signs_D = choose_stride_signs(nmode_D, negative_str, mixed_str);

    bool* subtensor_dims_A = choose_subtensor_dims(nmode_A, outer_nmode_A);
    bool* subtensor_dims_B = choose_subtensor_dims(nmode_B, outer_nmode_B);
    bool* subtensor_dims_C = choose_subtensor_dims(nmode_C, outer_nmode_C);
    bool* subtensor_dims_D = choose_subtensor_dims(nmode_D, outer_nmode_D);

    int64_t* outer_extents_A = calculate_outer_extents(outer_nmode_A, extents_A, subtensor_dims_A, lower_extents);
    int64_t* outer_extents_B = calculate_outer_extents(outer_nmode_B, extents_B, subtensor_dims_B, lower_extents);
    int64_t* outer_extents_C = calculate_outer_extents(outer_nmode_C, extents_C, subtensor_dims_C, lower_extents);
    int64_t* outer_extents_D = calculate_outer_extents(outer_nmode_D, extents_D, subtensor_dims_D, lower_extents);

    int64_t* offsets_A = calculate_offsets(nmode_A, outer_nmode_A, extents_A, outer_extents_A, subtensor_dims_A, lower_extents);
    int64_t* offsets_B = calculate_offsets(nmode_B, outer_nmode_B, extents_B, outer_extents_B, subtensor_dims_B, lower_extents);
    int64_t* offsets_C = calculate_offsets(nmode_C, outer_nmode_C, extents_C, outer_extents_C, subtensor_dims_C, lower_extents);
    int64_t* offsets_D = calculate_offsets(nmode_D, outer_nmode_D, extents_D, outer_extents_D, subtensor_dims_D, lower_extents);

    int64_t* strides_A = calculate_strides(nmode_A, outer_nmode_A, outer_extents_A, stride_signs_A, subtensor_dims_A);
    int64_t* strides_B = calculate_strides(nmode_B, outer_nmode_B, outer_extents_B, stride_signs_B, subtensor_dims_B);
    int64_t* strides_C = calculate_strides(nmode_C, outer_nmode_C, outer_extents_C, stride_signs_C, subtensor_dims_C);
    int64_t* strides_D = calculate_strides(nmode_D, outer_nmode_D, outer_extents_D, stride_signs_D, subtensor_dims_D);
    
    int64_t size_A = calculate_size(outer_nmode_A, outer_extents_A);
    int64_t size_B = calculate_size(outer_nmode_B, outer_extents_B);
    int64_t size_C = calculate_size(outer_nmode_C, outer_extents_C);
    int64_t size_D = calculate_size(outer_nmode_D, outer_extents_D);

    float* data_A = create_tensor_data_s(size_A);
    float* data_B = create_tensor_data_s(size_B);
    float* data_C = create_tensor_data_s(size_C);
    float* data_D = create_tensor_data_s(size_D);

    float* A = (float*)calculate_tensor_pointer(data_A, nmode_A, extents_A, offsets_A, strides_A, sizeof(float));
    float* B = (float*)calculate_tensor_pointer(data_B, nmode_B, extents_B, offsets_B, strides_B, sizeof(float));
    float* C = (float*)calculate_tensor_pointer(data_C, nmode_C, extents_C, offsets_C, strides_C, sizeof(float));
    float* D = (float*)calculate_tensor_pointer(data_D, nmode_D, extents_D, offsets_D, strides_D, sizeof(float));

    float alpha = rand_s();
    float beta = rand_s();

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

std::tuple<int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           int, int64_t*, int64_t*, double*, int64_t*,
           double, double,
           double*, double*, double*, double*,
           int64_t, int64_t, int64_t, int64_t> generate_contraction_d(int nmode_A = -1, int nmode_B = -1,
                                                        int nmode_D = randi(0, 4), int contractions = randi(0, 4),
                                                        int min_extent = 1, bool equal_extents = false,
                                                        bool lower_extents = false, bool lower_nmode = false,
                                                        bool negative_str = false, bool unique_idx = false,
                                                        bool repeated_idx = false, bool mixed_str = false)
{
    if (repeated_idx && nmode_D < 2)
    {
        nmode_D = randi(2, 4);
    }
    if (nmode_A == -1 && nmode_B == -1)
    {
        nmode_A = repeated_idx ? randi(1, nmode_D - 1) : randi(0, nmode_D);
        nmode_B = nmode_D - nmode_A;
        nmode_A = nmode_A + contractions;
        nmode_B = nmode_B + contractions;
    }
    else if (nmode_A == -1)
    {
        contractions = contractions > nmode_B ? (repeated_idx ? randi(0, nmode_B - 1) : randi(0, nmode_B)) : contractions;
        nmode_D = nmode_D < nmode_B - contractions ? nmode_B - contractions + (repeated_idx ? randi(1, 4) : randi(0, 4)) : nmode_D;
        nmode_A = contractions*2 + nmode_D - nmode_B;
    }
    else if (nmode_B == -1)
    {
        contractions = contractions > nmode_A ? (repeated_idx ? randi(0, nmode_A - 1) : randi(0, nmode_A)) : contractions;
        nmode_D = nmode_D < nmode_A - contractions ? nmode_A - contractions + (repeated_idx ? randi(1, 4) : randi(0, 4)) : nmode_D;
        nmode_B = contractions*2 + nmode_D - nmode_A;
    }
    else
    {
        contractions = contractions > std::min(nmode_A, nmode_B) ? randi(0, std::min(nmode_A, nmode_B)) : contractions;
        nmode_D = nmode_A + nmode_B - contractions * 2;
    }

    int unique_idx_A = unique_idx ? randi(1, 3) : 0;

    int unique_idx_B = unique_idx ? randi(1, 3) : 0;

    nmode_A += unique_idx_A;
    nmode_B += unique_idx_B;

    int repeated_idx_A = repeated_idx ? randi(1, 4) : 0;
    int repeated_idx_B = repeated_idx ? randi(1, 4) : 0;
    int repeated_idx_D = repeated_idx ? randi(1, 4) : 0;

    nmode_A += repeated_idx_A;
    nmode_B += repeated_idx_B;
    nmode_D += repeated_idx_D;
    
    int nmode_C = nmode_D;

    int64_t* idx_A = new int64_t[nmode_A];
    for (int i = 0; i < nmode_A - repeated_idx_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A - repeated_idx_A, std::default_random_engine());
    }

    
    int64_t* idx_B = new int64_t[nmode_B];
    int idx_contracted[contractions];
    for (int i = 0; i < contractions; i++)
    {
        idx_B[i] = idx_A[i];
        idx_contracted[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B - contractions - repeated_idx_B; i++)
    {
        idx_B[i + contractions] = 'a' + nmode_A - repeated_idx_A + i;
    }

    if (nmode_B > 0)
    {
        std::shuffle(idx_B, idx_B + nmode_B - repeated_idx_B, std::default_random_engine());
    }
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A - repeated_idx_A, std::default_random_engine());
    }

    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    int index = 0;
    int index_origin = 0;
    for (int i = 0; i < nmode_A - repeated_idx_A - unique_idx_A - contractions; i++)
    {
        for (int j = index_origin; j < nmode_A - repeated_idx_A; j++)
        {
            bool is_contracted = false;
            for (int k = 0; k < contractions; k++)
            {
                if (idx_A[j] == idx_contracted[k])
                {
                    is_contracted = true;
                    break;
                }
            }
            if (!is_contracted)
            {
                index_origin = j;
                break;
            }
        }
        idx_D[index] = idx_A[index_origin];
        index_origin++;
        index++;
    }
    index_origin = 0;
    for (int i = 0; i < nmode_B - repeated_idx_B - unique_idx_B - contractions; i++)
    {
        for (int j = index_origin; j < nmode_B - repeated_idx_B; j++)
        {
            bool is_contracted = false;
            for (int k = 0; k < contractions; k++)
            {
                if (idx_B[j] == idx_contracted[k])
                {
                    is_contracted = true;
                    break;
                }
            }
            if (!is_contracted)
            {
                index_origin = j;
                break;
            }
        }
        idx_D[index] = idx_B[index_origin];
        index_origin++;
        index++;
    }
    
    //Add repeated idx
    for (int i = 0; i < repeated_idx_A; i++)
    {
        idx_A[i + nmode_A - repeated_idx_A] = idx_A[randi(0, nmode_A - repeated_idx_A - 1)];
    }
    for (int i = 0; i < repeated_idx_B; i++)
    {
        idx_B[i + nmode_B - repeated_idx_B] = idx_B[randi(0, nmode_B - repeated_idx_B - 1)];
    }
    for (int i = 0; i < repeated_idx_D; i++)
    {
        idx_D[i + nmode_D - repeated_idx_D] = idx_D[randi(0, nmode_D - repeated_idx_D - 1)];
    }
    
    //Randomize order of idx
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }
    if (nmode_B > 0)
    {
        std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    }
    if (nmode_D > 0)
    {
        std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    }
    std::copy(idx_D, idx_D + nmode_D, idx_C);

    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    int64_t extent = randi(min_extent, 4);
    time_t time_seed = time(NULL);
    for (int i = 0; i < nmode_A; i++)
    {
        srand(time_seed * idx_A[i]);
        extents_A[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    for (int i = 0; i < nmode_B; i++)
    {
        srand(time_seed * idx_B[i]);
        extents_B[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    for (int i = 0; i < nmode_D; i++)
    {
        srand(time_seed * idx_D[i]);
        extents_D[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);

    int outer_nmode_A = lower_nmode ? nmode_A + randi(1, 4) : nmode_A;
    int outer_nmode_B = lower_nmode ? nmode_B + randi(1, 4) : nmode_B;
    int outer_nmode_C = lower_nmode ? nmode_C + randi(1, 4) : nmode_C;
    int outer_nmode_D = lower_nmode ? nmode_D + randi(1, 4) : nmode_D;

    int* stride_signs_A = choose_stride_signs(nmode_A, negative_str, mixed_str);
    int* stride_signs_B = choose_stride_signs(nmode_B, negative_str, mixed_str);
    int* stride_signs_C = choose_stride_signs(nmode_C, negative_str, mixed_str);
    int* stride_signs_D = choose_stride_signs(nmode_D, negative_str, mixed_str);

    bool* subtensor_dims_A = choose_subtensor_dims(nmode_A, outer_nmode_A);
    bool* subtensor_dims_B = choose_subtensor_dims(nmode_B, outer_nmode_B);
    bool* subtensor_dims_C = choose_subtensor_dims(nmode_C, outer_nmode_C);
    bool* subtensor_dims_D = choose_subtensor_dims(nmode_D, outer_nmode_D);

    int64_t* outer_extents_A = calculate_outer_extents(outer_nmode_A, extents_A, subtensor_dims_A, lower_extents);
    int64_t* outer_extents_B = calculate_outer_extents(outer_nmode_B, extents_B, subtensor_dims_B, lower_extents);
    int64_t* outer_extents_C = calculate_outer_extents(outer_nmode_C, extents_C, subtensor_dims_C, lower_extents);
    int64_t* outer_extents_D = calculate_outer_extents(outer_nmode_D, extents_D, subtensor_dims_D, lower_extents);

    int64_t* offsets_A = calculate_offsets(nmode_A, outer_nmode_A, extents_A, outer_extents_A, subtensor_dims_A, lower_extents);
    int64_t* offsets_B = calculate_offsets(nmode_B, outer_nmode_B, extents_B, outer_extents_B, subtensor_dims_B, lower_extents);
    int64_t* offsets_C = calculate_offsets(nmode_C, outer_nmode_C, extents_C, outer_extents_C, subtensor_dims_C, lower_extents);
    int64_t* offsets_D = calculate_offsets(nmode_D, outer_nmode_D, extents_D, outer_extents_D, subtensor_dims_D, lower_extents);

    int64_t* strides_A = calculate_strides(nmode_A, outer_nmode_A, outer_extents_A, stride_signs_A, subtensor_dims_A);
    int64_t* strides_B = calculate_strides(nmode_B, outer_nmode_B, outer_extents_B, stride_signs_B, subtensor_dims_B);
    int64_t* strides_C = calculate_strides(nmode_C, outer_nmode_C, outer_extents_C, stride_signs_C, subtensor_dims_C);
    int64_t* strides_D = calculate_strides(nmode_D, outer_nmode_D, outer_extents_D, stride_signs_D, subtensor_dims_D);
    
    int64_t size_A = calculate_size(outer_nmode_A, outer_extents_A);
    int64_t size_B = calculate_size(outer_nmode_B, outer_extents_B);
    int64_t size_C = calculate_size(outer_nmode_C, outer_extents_C);
    int64_t size_D = calculate_size(outer_nmode_D, outer_extents_D);

    double* data_A = create_tensor_data_d(size_A);
    double* data_B = create_tensor_data_d(size_B);
    double* data_C = create_tensor_data_d(size_C);
    double* data_D = create_tensor_data_d(size_D);

    double* A = (double*)calculate_tensor_pointer(data_A, nmode_A, extents_A, offsets_A, strides_A, sizeof(double));
    double* B = (double*)calculate_tensor_pointer(data_B, nmode_B, extents_B, offsets_B, strides_B, sizeof(double));
    double* C = (double*)calculate_tensor_pointer(data_C, nmode_C, extents_C, offsets_C, strides_C, sizeof(double));
    double* D = (double*)calculate_tensor_pointer(data_D, nmode_D, extents_D, offsets_D, strides_D, sizeof(double));

    double alpha = rand_d();
    double beta = rand_d();

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

std::tuple<int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<float>*, int64_t*,
           std::complex<float>, std::complex<float>,
           std::complex<float>*, std::complex<float>*, std::complex<float>*, std::complex<float>*,
           int64_t, int64_t, int64_t, int64_t> generate_contraction_c(int nmode_A = -1, int nmode_B = -1,
                                                        int nmode_D = randi(0, 4), int contractions = randi(0, 4),
                                                        int min_extent = 1, bool equal_extents = false,
                                                        bool lower_extents = false, bool lower_nmode = false,
                                                        bool negative_str = false, bool unique_idx = false,
                                                        bool repeated_idx = false, bool mixed_str = false)
{
    if (repeated_idx && nmode_D < 2)
    {
        nmode_D = randi(2, 4);
    }
    if (nmode_A == -1 && nmode_B == -1)
    {
        nmode_A = repeated_idx ? randi(1, nmode_D - 1) : randi(0, nmode_D);
        nmode_B = nmode_D - nmode_A;
        nmode_A = nmode_A + contractions;
        nmode_B = nmode_B + contractions;
    }
    else if (nmode_A == -1)
    {
        contractions = contractions > nmode_B ? (repeated_idx ? randi(0, nmode_B - 1) : randi(0, nmode_B)) : contractions;
        nmode_D = nmode_D < nmode_B - contractions ? nmode_B - contractions + (repeated_idx ? randi(1, 4) : randi(0, 4)) : nmode_D;
        nmode_A = contractions*2 + nmode_D - nmode_B;
    }
    else if (nmode_B == -1)
    {
        contractions = contractions > nmode_A ? (repeated_idx ? randi(0, nmode_A - 1) : randi(0, nmode_A)) : contractions;
        nmode_D = nmode_D < nmode_A - contractions ? nmode_A - contractions + (repeated_idx ? randi(1, 4) : randi(0, 4)) : nmode_D;
        nmode_B = contractions*2 + nmode_D - nmode_A;
    }
    else
    {
        contractions = contractions > std::min(nmode_A, nmode_B) ? randi(0, std::min(nmode_A, nmode_B)) : contractions;
        nmode_D = nmode_A + nmode_B - contractions * 2;
    }

    int unique_idx_A = unique_idx ? randi(1, 3) : 0;

    int unique_idx_B = unique_idx ? randi(1, 3) : 0;

    nmode_A += unique_idx_A;
    nmode_B += unique_idx_B;

    int repeated_idx_A = repeated_idx ? randi(1, 4) : 0;
    int repeated_idx_B = repeated_idx ? randi(1, 4) : 0;
    int repeated_idx_D = repeated_idx ? randi(1, 4) : 0;

    nmode_A += repeated_idx_A;
    nmode_B += repeated_idx_B;
    nmode_D += repeated_idx_D;
    
    int nmode_C = nmode_D;

    int64_t* idx_A = new int64_t[nmode_A];
    for (int i = 0; i < nmode_A - repeated_idx_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A - repeated_idx_A, std::default_random_engine());
    }

    
    int64_t* idx_B = new int64_t[nmode_B];
    int idx_contracted[contractions];
    for (int i = 0; i < contractions; i++)
    {
        idx_B[i] = idx_A[i];
        idx_contracted[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B - contractions - repeated_idx_B; i++)
    {
        idx_B[i + contractions] = 'a' + nmode_A - repeated_idx_A + i;
    }

    if (nmode_B > 0)
    {
        std::shuffle(idx_B, idx_B + nmode_B - repeated_idx_B, std::default_random_engine());
    }
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A - repeated_idx_A, std::default_random_engine());
    }

    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    int index = 0;
    int index_origin = 0;
    for (int i = 0; i < nmode_A - repeated_idx_A - unique_idx_A - contractions; i++)
    {
        for (int j = index_origin; j < nmode_A - repeated_idx_A; j++)
        {
            bool is_contracted = false;
            for (int k = 0; k < contractions; k++)
            {
                if (idx_A[j] == idx_contracted[k])
                {
                    is_contracted = true;
                    break;
                }
            }
            if (!is_contracted)
            {
                index_origin = j;
                break;
            }
        }
        idx_D[index] = idx_A[index_origin];
        index_origin++;
        index++;
    }
    index_origin = 0;
    for (int i = 0; i < nmode_B - repeated_idx_B - unique_idx_B - contractions; i++)
    {
        for (int j = index_origin; j < nmode_B - repeated_idx_B; j++)
        {
            bool is_contracted = false;
            for (int k = 0; k < contractions; k++)
            {
                if (idx_B[j] == idx_contracted[k])
                {
                    is_contracted = true;
                    break;
                }
            }
            if (!is_contracted)
            {
                index_origin = j;
                break;
            }
        }
        idx_D[index] = idx_B[index_origin];
        index_origin++;
        index++;
    }
    
    //Add repeated idx
    for (int i = 0; i < repeated_idx_A; i++)
    {
        idx_A[i + nmode_A - repeated_idx_A] = idx_A[randi(0, nmode_A - repeated_idx_A - 1)];
    }
    for (int i = 0; i < repeated_idx_B; i++)
    {
        idx_B[i + nmode_B - repeated_idx_B] = idx_B[randi(0, nmode_B - repeated_idx_B - 1)];
    }
    for (int i = 0; i < repeated_idx_D; i++)
    {
        idx_D[i + nmode_D - repeated_idx_D] = idx_D[randi(0, nmode_D - repeated_idx_D - 1)];
    }
    
    //Randomize order of idx
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }
    if (nmode_B > 0)
    {
        std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    }
    if (nmode_D > 0)
    {
        std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    }
    std::copy(idx_D, idx_D + nmode_D, idx_C);

    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    int64_t extent = randi(min_extent, 4);
    time_t time_seed = time(NULL);
    for (int i = 0; i < nmode_A; i++)
    {
        srand(time_seed * idx_A[i]);
        extents_A[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    for (int i = 0; i < nmode_B; i++)
    {
        srand(time_seed * idx_B[i]);
        extents_B[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    for (int i = 0; i < nmode_D; i++)
    {
        srand(time_seed * idx_D[i]);
        extents_D[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);

    int outer_nmode_A = lower_nmode ? nmode_A + randi(1, 4) : nmode_A;
    int outer_nmode_B = lower_nmode ? nmode_B + randi(1, 4) : nmode_B;
    int outer_nmode_C = lower_nmode ? nmode_C + randi(1, 4) : nmode_C;
    int outer_nmode_D = lower_nmode ? nmode_D + randi(1, 4) : nmode_D;

    int* stride_signs_A = choose_stride_signs(nmode_A, negative_str, mixed_str);
    int* stride_signs_B = choose_stride_signs(nmode_B, negative_str, mixed_str);
    int* stride_signs_C = choose_stride_signs(nmode_C, negative_str, mixed_str);
    int* stride_signs_D = choose_stride_signs(nmode_D, negative_str, mixed_str);

    bool* subtensor_dims_A = choose_subtensor_dims(nmode_A, outer_nmode_A);
    bool* subtensor_dims_B = choose_subtensor_dims(nmode_B, outer_nmode_B);
    bool* subtensor_dims_C = choose_subtensor_dims(nmode_C, outer_nmode_C);
    bool* subtensor_dims_D = choose_subtensor_dims(nmode_D, outer_nmode_D);

    int64_t* outer_extents_A = calculate_outer_extents(outer_nmode_A, extents_A, subtensor_dims_A, lower_extents);
    int64_t* outer_extents_B = calculate_outer_extents(outer_nmode_B, extents_B, subtensor_dims_B, lower_extents);
    int64_t* outer_extents_C = calculate_outer_extents(outer_nmode_C, extents_C, subtensor_dims_C, lower_extents);
    int64_t* outer_extents_D = calculate_outer_extents(outer_nmode_D, extents_D, subtensor_dims_D, lower_extents);

    int64_t* offsets_A = calculate_offsets(nmode_A, outer_nmode_A, extents_A, outer_extents_A, subtensor_dims_A, lower_extents);
    int64_t* offsets_B = calculate_offsets(nmode_B, outer_nmode_B, extents_B, outer_extents_B, subtensor_dims_B, lower_extents);
    int64_t* offsets_C = calculate_offsets(nmode_C, outer_nmode_C, extents_C, outer_extents_C, subtensor_dims_C, lower_extents);
    int64_t* offsets_D = calculate_offsets(nmode_D, outer_nmode_D, extents_D, outer_extents_D, subtensor_dims_D, lower_extents);

    int64_t* strides_A = calculate_strides(nmode_A, outer_nmode_A, outer_extents_A, stride_signs_A, subtensor_dims_A);
    int64_t* strides_B = calculate_strides(nmode_B, outer_nmode_B, outer_extents_B, stride_signs_B, subtensor_dims_B);
    int64_t* strides_C = calculate_strides(nmode_C, outer_nmode_C, outer_extents_C, stride_signs_C, subtensor_dims_C);
    int64_t* strides_D = calculate_strides(nmode_D, outer_nmode_D, outer_extents_D, stride_signs_D, subtensor_dims_D);
    
    int64_t size_A = calculate_size(outer_nmode_A, outer_extents_A);
    int64_t size_B = calculate_size(outer_nmode_B, outer_extents_B);
    int64_t size_C = calculate_size(outer_nmode_C, outer_extents_C);
    int64_t size_D = calculate_size(outer_nmode_D, outer_extents_D);

    std::complex<float>* data_A = create_tensor_data_c(size_A);
    std::complex<float>* data_B = create_tensor_data_c(size_B);
    std::complex<float>* data_C = create_tensor_data_c(size_C);
    std::complex<float>* data_D = create_tensor_data_c(size_D);

    std::complex<float>* A = (std::complex<float>*)calculate_tensor_pointer(data_A, nmode_A, extents_A, offsets_A, strides_A, sizeof(std::complex<float>));
    std::complex<float>* B = (std::complex<float>*)calculate_tensor_pointer(data_B, nmode_B, extents_B, offsets_B, strides_B, sizeof(std::complex<float>));
    std::complex<float>* C = (std::complex<float>*)calculate_tensor_pointer(data_C, nmode_C, extents_C, offsets_C, strides_C, sizeof(std::complex<float>));
    std::complex<float>* D = (std::complex<float>*)calculate_tensor_pointer(data_D, nmode_D, extents_D, offsets_D, strides_D, sizeof(std::complex<float>));

    std::complex<float> alpha = rand_c();
    std::complex<float> beta = rand_c();

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

std::tuple<int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           int, int64_t*, int64_t*, std::complex<double>*, int64_t*,
           std::complex<double>, std::complex<double>,
           std::complex<double>*, std::complex<double>*, std::complex<double>*, std::complex<double>*,
           int64_t, int64_t, int64_t, int64_t> generate_contraction_z(int nmode_A = -1, int nmode_B = -1,
                                                        int nmode_D = randi(0, 4), int contractions = randi(0, 4),
                                                        int min_extent = 1, bool equal_extents = false,
                                                        bool lower_extents = false, bool lower_nmode = false,
                                                        bool negative_str = false, bool unique_idx = false,
                                                        bool repeated_idx = false, bool mixed_str = false)
{
    if (repeated_idx && nmode_D < 2)
    {
        nmode_D = randi(2, 4);
    }
    if (nmode_A == -1 && nmode_B == -1)
    {
        nmode_A = repeated_idx ? randi(1, nmode_D - 1) : randi(0, nmode_D);
        nmode_B = nmode_D - nmode_A;
        nmode_A = nmode_A + contractions;
        nmode_B = nmode_B + contractions;
    }
    else if (nmode_A == -1)
    {
        contractions = contractions > nmode_B ? (repeated_idx ? randi(0, nmode_B - 1) : randi(0, nmode_B)) : contractions;
        nmode_D = nmode_D < nmode_B - contractions ? nmode_B - contractions + (repeated_idx ? randi(1, 4) : randi(0, 4)) : nmode_D;
        nmode_A = contractions*2 + nmode_D - nmode_B;
    }
    else if (nmode_B == -1)
    {
        contractions = contractions > nmode_A ? (repeated_idx ? randi(0, nmode_A - 1) : randi(0, nmode_A)) : contractions;
        nmode_D = nmode_D < nmode_A - contractions ? nmode_A - contractions + (repeated_idx ? randi(1, 4) : randi(0, 4)) : nmode_D;
        nmode_B = contractions*2 + nmode_D - nmode_A;
    }
    else
    {
        contractions = contractions > std::min(nmode_A, nmode_B) ? randi(0, std::min(nmode_A, nmode_B)) : contractions;
        nmode_D = nmode_A + nmode_B - contractions * 2;
    }

    int unique_idx_A = unique_idx ? randi(1, 3) : 0;

    int unique_idx_B = unique_idx ? randi(1, 3) : 0;

    nmode_A += unique_idx_A;
    nmode_B += unique_idx_B;

    int repeated_idx_A = repeated_idx ? randi(1, 4) : 0;
    int repeated_idx_B = repeated_idx ? randi(1, 4) : 0;
    int repeated_idx_D = repeated_idx ? randi(1, 4) : 0;

    nmode_A += repeated_idx_A;
    nmode_B += repeated_idx_B;
    nmode_D += repeated_idx_D;
    
    int nmode_C = nmode_D;

    int64_t* idx_A = new int64_t[nmode_A];
    for (int i = 0; i < nmode_A - repeated_idx_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A - repeated_idx_A, std::default_random_engine());
    }

    
    int64_t* idx_B = new int64_t[nmode_B];
    int idx_contracted[contractions];
    for (int i = 0; i < contractions; i++)
    {
        idx_B[i] = idx_A[i];
        idx_contracted[i] = idx_A[i];
    }
    for (int i = 0; i < nmode_B - contractions - repeated_idx_B; i++)
    {
        idx_B[i + contractions] = 'a' + nmode_A - repeated_idx_A + i;
    }

    if (nmode_B > 0)
    {
        std::shuffle(idx_B, idx_B + nmode_B - repeated_idx_B, std::default_random_engine());
    }
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A - repeated_idx_A, std::default_random_engine());
    }

    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    int index = 0;
    int index_origin = 0;
    for (int i = 0; i < nmode_A - repeated_idx_A - unique_idx_A - contractions; i++)
    {
        for (int j = index_origin; j < nmode_A - repeated_idx_A; j++)
        {
            bool is_contracted = false;
            for (int k = 0; k < contractions; k++)
            {
                if (idx_A[j] == idx_contracted[k])
                {
                    is_contracted = true;
                    break;
                }
            }
            if (!is_contracted)
            {
                index_origin = j;
                break;
            }
        }
        idx_D[index] = idx_A[index_origin];
        index_origin++;
        index++;
    }
    index_origin = 0;
    for (int i = 0; i < nmode_B - repeated_idx_B - unique_idx_B - contractions; i++)
    {
        for (int j = index_origin; j < nmode_B - repeated_idx_B; j++)
        {
            bool is_contracted = false;
            for (int k = 0; k < contractions; k++)
            {
                if (idx_B[j] == idx_contracted[k])
                {
                    is_contracted = true;
                    break;
                }
            }
            if (!is_contracted)
            {
                index_origin = j;
                break;
            }
        }
        idx_D[index] = idx_B[index_origin];
        index_origin++;
        index++;
    }
    
    //Add repeated idx
    for (int i = 0; i < repeated_idx_A; i++)
    {
        idx_A[i + nmode_A - repeated_idx_A] = idx_A[randi(0, nmode_A - repeated_idx_A - 1)];
    }
    for (int i = 0; i < repeated_idx_B; i++)
    {
        idx_B[i + nmode_B - repeated_idx_B] = idx_B[randi(0, nmode_B - repeated_idx_B - 1)];
    }
    for (int i = 0; i < repeated_idx_D; i++)
    {
        idx_D[i + nmode_D - repeated_idx_D] = idx_D[randi(0, nmode_D - repeated_idx_D - 1)];
    }
    
    //Randomize order of idx
    if (nmode_A > 0)
    {
        std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    }
    if (nmode_B > 0)
    {
        std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    }
    if (nmode_D > 0)
    {
        std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    }
    std::copy(idx_D, idx_D + nmode_D, idx_C);

    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    int64_t extent = randi(min_extent, 4);
    time_t time_seed = time(NULL);
    for (int i = 0; i < nmode_A; i++)
    {
        srand(time_seed * idx_A[i]);
        extents_A[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    for (int i = 0; i < nmode_B; i++)
    {
        srand(time_seed * idx_B[i]);
        extents_B[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    for (int i = 0; i < nmode_D; i++)
    {
        srand(time_seed * idx_D[i]);
        extents_D[i] = equal_extents ? extent : randi(min_extent, 4);
    }
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);

    int outer_nmode_A = lower_nmode ? nmode_A + randi(1, 4) : nmode_A;
    int outer_nmode_B = lower_nmode ? nmode_B + randi(1, 4) : nmode_B;
    int outer_nmode_C = lower_nmode ? nmode_C + randi(1, 4) : nmode_C;
    int outer_nmode_D = lower_nmode ? nmode_D + randi(1, 4) : nmode_D;

    int* stride_signs_A = choose_stride_signs(nmode_A, negative_str, mixed_str);
    int* stride_signs_B = choose_stride_signs(nmode_B, negative_str, mixed_str);
    int* stride_signs_C = choose_stride_signs(nmode_C, negative_str, mixed_str);
    int* stride_signs_D = choose_stride_signs(nmode_D, negative_str, mixed_str);

    bool* subtensor_dims_A = choose_subtensor_dims(nmode_A, outer_nmode_A);
    bool* subtensor_dims_B = choose_subtensor_dims(nmode_B, outer_nmode_B);
    bool* subtensor_dims_C = choose_subtensor_dims(nmode_C, outer_nmode_C);
    bool* subtensor_dims_D = choose_subtensor_dims(nmode_D, outer_nmode_D);

    int64_t* outer_extents_A = calculate_outer_extents(outer_nmode_A, extents_A, subtensor_dims_A, lower_extents);
    int64_t* outer_extents_B = calculate_outer_extents(outer_nmode_B, extents_B, subtensor_dims_B, lower_extents);
    int64_t* outer_extents_C = calculate_outer_extents(outer_nmode_C, extents_C, subtensor_dims_C, lower_extents);
    int64_t* outer_extents_D = calculate_outer_extents(outer_nmode_D, extents_D, subtensor_dims_D, lower_extents);

    int64_t* offsets_A = calculate_offsets(nmode_A, outer_nmode_A, extents_A, outer_extents_A, subtensor_dims_A, lower_extents);
    int64_t* offsets_B = calculate_offsets(nmode_B, outer_nmode_B, extents_B, outer_extents_B, subtensor_dims_B, lower_extents);
    int64_t* offsets_C = calculate_offsets(nmode_C, outer_nmode_C, extents_C, outer_extents_C, subtensor_dims_C, lower_extents);
    int64_t* offsets_D = calculate_offsets(nmode_D, outer_nmode_D, extents_D, outer_extents_D, subtensor_dims_D, lower_extents);

    int64_t* strides_A = calculate_strides(nmode_A, outer_nmode_A, outer_extents_A, stride_signs_A, subtensor_dims_A);
    int64_t* strides_B = calculate_strides(nmode_B, outer_nmode_B, outer_extents_B, stride_signs_B, subtensor_dims_B);
    int64_t* strides_C = calculate_strides(nmode_C, outer_nmode_C, outer_extents_C, stride_signs_C, subtensor_dims_C);
    int64_t* strides_D = calculate_strides(nmode_D, outer_nmode_D, outer_extents_D, stride_signs_D, subtensor_dims_D);
    
    int64_t size_A = calculate_size(outer_nmode_A, outer_extents_A);
    int64_t size_B = calculate_size(outer_nmode_B, outer_extents_B);
    int64_t size_C = calculate_size(outer_nmode_C, outer_extents_C);
    int64_t size_D = calculate_size(outer_nmode_D, outer_extents_D);

    std::complex<double>* data_A = create_tensor_data_z(size_A);
    std::complex<double>* data_B = create_tensor_data_z(size_B);
    std::complex<double>* data_C = create_tensor_data_z(size_C);
    std::complex<double>* data_D = create_tensor_data_z(size_D);

    std::complex<double>* A = (std::complex<double>*)calculate_tensor_pointer(data_A, nmode_A, extents_A, offsets_A, strides_A, sizeof(std::complex<double>));
    std::complex<double>* B = (std::complex<double>*)calculate_tensor_pointer(data_B, nmode_B, extents_B, offsets_B, strides_B, sizeof(std::complex<double>));
    std::complex<double>* C = (std::complex<double>*)calculate_tensor_pointer(data_C, nmode_C, extents_C, offsets_C, strides_C, sizeof(std::complex<double>));
    std::complex<double>* D = (std::complex<double>*)calculate_tensor_pointer(data_D, nmode_D, extents_D, offsets_D, strides_D, sizeof(std::complex<double>));
    std::complex<double> zmi{1.0e-14,1.0e-14}; //+ 2I
    std::complex<double> zma{1.0e-1,1.0e-1};
    std::complex<double> alpha = rand_z(zmi,zma);
    std::complex<double> beta = rand_z(zmi,zma);

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

int* choose_stride_signs(int nmode, bool negative_str, bool mixed_str)
{
    int* stride_signs = new int[nmode];
    int negative_str_count = 0;

    for (size_t i = 0; i < nmode; i++)
    {
        if (negative_str)
        {
            stride_signs[i] = -1;
        }
        else if (mixed_str)
        {
            if ((randi(0, 1) == 0 && negative_str_count < nmode/2) || (negative_str_count < (i - nmode/2)))
            {
                stride_signs[i] = -1;
            }
            else
            {
                stride_signs[i] = 1;
            }
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
        if ((rand_s(0, 1) < (float)nmode/(float)outer_nmode || outer_nmode - i == nmode - idx) && nmode - idx > 0)
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
            int extension = randi(1, 4);
            outer_extents[i] = lower_extents ? extents[idx] + extension : extents[idx];
            idx++;
        }
        else
        {
            outer_extents[i] = lower_extents ? randi(1, 8) : randi(1, 4);
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
            offsets[idx] = lower_extents && outer_extents[i] - extents[idx] > 0 ? randi(0, outer_extents[i] - extents[idx]) : 0;
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

int64_t* calculate_simple_strides(int nmode, int64_t* extents)
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

float* create_tensor_data_s(int64_t size)
{
    float* data = new float[size];
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand_s();
    }
    return data;
}

double* create_tensor_data_d(int64_t size)
{
    double* data = new double[size];
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand_d();
    }
    return data;
}

std::complex<float>* create_tensor_data_c(int64_t size)
{
    std::complex<float>* data = new std::complex<float>[size];
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand_c();
    }
    return data;
}

std::complex<double>* create_tensor_data_z(int64_t size)
{
    std::complex<double> zmi{1.0e-14,1.0e-14}; //+ 2I
    std::complex<double> zma{1.0e-1,1.0e-1};

    std::complex<double>* data = new std::complex<double>[size];
    for (size_t i = 0; i < size; i++)
    {
        data[i] = rand_z(zmi, zma);
    }
    return data;
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

std::tuple<float*, float*> copy_tensor_data_s(int64_t size, float* data, float* pointer)
{
    float* new_data = new float[size];
    std::copy(data, data + size, new_data);
    float* new_pointer = (float*)((intptr_t)new_data + (intptr_t)pointer - (intptr_t)data);
    return {new_pointer, new_data};
}

std::tuple<double*, double*> copy_tensor_data_d(int64_t size, double* data, double* pointer)
{
    double* new_data = new double[size];
    std::copy(data, data + size, new_data);
    double* new_pointer = (double*)((intptr_t)new_data + (intptr_t)pointer - (intptr_t)data);
    return {new_pointer, new_data};
}

std::tuple<std::complex<float>*, std::complex<float>*> copy_tensor_data_c(int64_t size, std::complex<float>* data, std::complex<float>* pointer)
{
    std::complex<float>* new_data = new std::complex<float>[size];
    std::copy(data, data + size, new_data);
    std::complex<float>* new_pointer = (std::complex<float>*)((intptr_t)new_data + (intptr_t)pointer - (intptr_t)data);
    return {new_pointer, new_data};
}

std::tuple<std::complex<double>*, std::complex<double>*> copy_tensor_data_z(int64_t size, std::complex<double>* data, std::complex<double>* pointer)
{
    std::complex<double>* new_data = new std::complex<double>[size];
    std::copy(data, data + size, new_data);
    std::complex<double>* new_pointer = (std::complex<double>*)((intptr_t)new_data + (intptr_t)pointer - (intptr_t)data);
    return {new_pointer, new_data};
}

float* copy_tensor_data_s(int size, float* data)
{
    float* dataA = new float[size];
    std::copy(data, data + size, dataA);
    return dataA;
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

std::string str(bool b)
{
    return b ? "true" : "false";
}

int randi(int min, int max)
{
    return rand() % (max - min + 1) + min;
}

float rand_s(float min, float max)
{
    return min + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(max-min)));
}

double rand_d(double min, double max)
{
    return min + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(max-min)));
}

int random_choice(int size, int* choices)
{
    return choices[randi(0, size - 1)];
}

std::complex<float> rand_c(std::complex<float> min, std::complex<float> max)
{
    return std::complex<float>(min.real() + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(max.real()-min.real()))), min.imag() + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/(max.imag()-min.imag()))));
}

std::complex<double> rand_z(std::complex<double> min, std::complex<double> max)
{
    return std::complex<double>(min.real() + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(max.real()-min.real()))), min.imag() + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/(max.imag()-min.imag()))));
}

float rand_s()
{
    return (rand() + static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * (rand() % 2 == 0 ? 1 : -1);
}

double rand_d()
{
    return (rand() + static_cast <double> (rand()) / static_cast <double> (RAND_MAX)) * (rand() % 2 == 0 ? 1 : -1);
}

std::complex<float> rand_c()
{
    return std::complex<float>(rand_s(), rand_s());
}

std::complex<double> rand_z()
{
    return std::complex<double>(rand_d(), rand_d());
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

void print_tensor_s(int nmode, int64_t* extents, int64_t* strides, float* data)
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
    int size = calculate_size(nmode, extents);
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

void print_tensor_d(int nmode, int64_t* extents, int64_t* strides, double* data)
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

void print_tensor_c(int nmode, int64_t* extents, int64_t* strides, std::complex<float>* data)
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

void print_tensor_z(int nmode, int64_t* extents, int64_t* strides, std::complex<double>* data)
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
    int nmode = randi(0, 4);
    int64_t* extents = new int64_t[nmode];
    int64_t* strides = new int64_t[nmode];
    int size = 1;
    for (int i = 0; i < nmode; i++)
    {
        extents[i] = randi(1, 4);
        size *= extents[i];
    }
    if (nmode > 0)
    {
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
        A[i] = rand_s(0, 1);
        B[i] = rand_s(0, 1);
        C[i] = rand_s(0, 1);
        D[i] = rand_s(0, 1);
    }

    float alpha = rand_s(0, 1);
    float beta = rand_s(0, 1);

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
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
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

bool test_contraction()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_contraction_s();

    auto [E, data_E] = copy_tensor_data_s(size_D, data_D, D);

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
          size_A, size_B, size_C, size_D] = generate_contraction_s();

    auto [E, data_E] = copy_tensor_data_s(size_D, data_D, D);

    auto [F, data_F] = copy_tensor_data_s(size_D, data_D, D);

    auto [G, data_G] = copy_tensor_data_s(size_D, data_D, D);

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
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, E, 0, idx_D,
                   alpha, beta);

    TAPP_execute_product(planBA, exec, &status, (void*)&alpha, (void*)B, (void*)A, (void*)&beta, (void*)C, (void*)F);

    run_tblis_mult_s(nmode_B, extents_B, strides_B, B, 0, idx_B,
                   nmode_A, extents_A, strides_A, A, 0, idx_A,
                   nmode_C, extents_C, strides_C, C, 0, idx_D,
                   nmode_D, extents_D, strides_D, G, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D) && compare_tensors_s(data_F, data_G, size_D) && compare_tensors_s(data_D, data_F, size_D);
    
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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(2, 4));
          
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);

    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    TAPP_tensor_product plan;
    TAPP_handle handle;
    create_handle(&handle);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);
    
    bool result = true;

    for (int i = 0; i < nmode_D; i++)
    {
        TAPP_tensor_info info_C;
        TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);
        TAPP_tensor_info info_D;
        TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);
        TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
        TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);
        run_tblis_mult_s(nmode_A, extents_A, strides_A, A, 0, idx_A,
                    nmode_B, extents_B, strides_B, B, 0, idx_B,
                    nmode_C, extents_C, strides_C, C, 0, idx_D,
                    nmode_D, extents_D, strides_D, E, 0, idx_D,
                    alpha, beta);
        
        result = result && compare_tensors_s(data_D, data_E, size_D);

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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);

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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), 0);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, 0);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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
          size_A, size_B, size_C, size_D] = generate_contraction_s(0);//2,2,0,2);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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
          size_A, size_B, size_C, size_D] = generate_contraction_s(1);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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

bool test_subtensor_same_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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

bool test_subtensor_lower_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, true, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, false, false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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

bool test_negative_strides_subtensor_same_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, true, false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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

bool test_negative_strides_subtensor_lower_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, true, true, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, false, false, false, false, false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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

bool test_mixed_strides_subtensor_same_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, true, false, false, false, false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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

bool test_mixed_strides_subtensor_lower_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, true, true, false, false, false, true);
    
    auto[E, data_E] = copy_tensor_data_s(size_D, data_D, D);
    
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
          size_A, size_B, size_C, size_D] = generate_contraction_d();

    auto [E, data_E] = copy_tensor_data_d(size_D, data_D, D);

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
          size_A, size_B, size_C, size_D] = generate_contraction_c();

    auto [E, data_E] = copy_tensor_data_c(size_D, data_D, D);

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
          size_A, size_B, size_C, size_D] = generate_contraction_z(2,2,0,2);//2,2,0,2);

    auto [E, data_E] = copy_tensor_data_z(size_D, data_D, D);

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

    int terr = TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)A, (void*)B, (void*)&beta, (void*)C, (void*)D);

    run_tblis_mult_z(nmode_A, extents_A, strides_A, A, op_A, idx_A,
                     nmode_B, extents_B, strides_B, B, op_B, idx_B,
                     nmode_C, extents_C, strides_C, C, op_C, idx_D,
                     nmode_D, extents_D, strides_D, E, op_D, idx_D,
                     alpha, beta);
    // std::complex<double> zma = 1.0+1.0e-12;
    // data_D[0] = data_D[0]*zma;
    bool result = compare_tensors_z(data_D, data_E, size_D);

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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(1, 4));

    auto [E, data_E] = copy_tensor_data_s(size_D, data_D, D);

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

bool test_unique_idx()
{
    auto [nmode_A, extents_A, strides_A, A, idx_A,
          nmode_B, extents_B, strides_B, B, idx_B,
          nmode_C, extents_C, strides_C, C, idx_C,
          nmode_D, extents_D, strides_D, D, idx_D,
          alpha, beta,
          data_A, data_B, data_C, data_D,
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, false, false, false, true, false);

    auto [E, data_E] = copy_tensor_data_s(size_D, data_D, D);

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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(0, 4), randi(0, 4), 1, false, false, false, false, false, true);

    auto [E, data_E] = copy_tensor_data_s(size_D, data_D, D);

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
    int nmode_A = randi(1, 4);
    int nmode_B = nmode_A + randi(1, 3);
    int nmode_D = nmode_B;
    int nmode_C = nmode_D;

    int64_t* idx_A = new int64_t[nmode_A];
    int64_t* idx_B = new int64_t[nmode_B];
    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    for (int i = 0; i < nmode_D; i++)
    {
        idx_D[i] = 'a' + i;
    }
    std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());
    
    std::copy(idx_D, idx_D + nmode_A, idx_A);
    std::copy(idx_D, idx_D + nmode_B, idx_B);
    
    std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());

    std::copy(idx_D, idx_D + nmode_C, idx_C);
    
    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    time_t time_seed = time(NULL);
    for (int i = 0; i < nmode_A; i++)
    {
        srand(time_seed + idx_A[i]);
        extents_A[i] = randi(1, 4);
    }
    for (int i = 0; i < nmode_B; i++)
    {
        srand(time_seed + idx_B[i]);
        extents_B[i] = randi(1, 4);
    }
    for (int i = 0; i < nmode_D; i++)
    {
        srand(time_seed + idx_D[i]);
        extents_D[i] = randi(1, 4);
    }    
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);
    
    int64_t* strides_A = calculate_simple_strides(nmode_A, extents_A);
    int64_t* strides_B = calculate_simple_strides(nmode_B, extents_B);
    int64_t* strides_C = calculate_simple_strides(nmode_C, extents_C);
    int64_t* strides_D = calculate_simple_strides(nmode_D, extents_D);

    int size_A = calculate_size(nmode_A, extents_A);
    int size_B = calculate_size(nmode_B, extents_B);
    int size_C = calculate_size(nmode_C, extents_C);
    int size_D = calculate_size(nmode_D, extents_D);
    
    float* data_A = create_tensor_data_s(size_A);
    float* data_B = create_tensor_data_s(size_B);
    float* data_C = create_tensor_data_s(size_C);
    float* data_D = create_tensor_data_s(size_D);
    
    float* data_E = copy_tensor_data_s(size_D, data_D);

    float alpha = rand_s();
    float beta = rand_s();

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

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)data_A, (void*)data_B, (void*)&beta, (void*)data_C, (void*)data_D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, data_A, 0, idx_A,
                   nmode_B, extents_B, strides_B, data_B, 0, idx_B,
                   nmode_C, extents_C, strides_C, data_C, 0, idx_D,
                   nmode_D, extents_D, strides_D, data_E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

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
    int nmode_D = randi(1, 4);
    int nmode_A = nmode_D + randi(1, 3);
    int nmode_B = nmode_A;
    int nmode_C = nmode_D;

    int64_t* idx_A = new int64_t[nmode_A];
    int64_t* idx_B = new int64_t[nmode_B];
    int64_t* idx_C = new int64_t[nmode_C];
    int64_t* idx_D = new int64_t[nmode_D];
    for (int i = 0; i < nmode_A; i++)
    {
        idx_A[i] = 'a' + i;
    }
    std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    
    std::copy(idx_A, idx_A + nmode_B, idx_B);
    std::copy(idx_A, idx_A + nmode_D, idx_D);
    
    std::shuffle(idx_A, idx_A + nmode_A, std::default_random_engine());
    std::shuffle(idx_B, idx_B + nmode_B, std::default_random_engine());
    std::shuffle(idx_D, idx_D + nmode_D, std::default_random_engine());

    std::copy(idx_D, idx_D + nmode_C, idx_C);
    
    int64_t* extents_A = new int64_t[nmode_A];
    int64_t* extents_B = new int64_t[nmode_B];
    int64_t* extents_D = new int64_t[nmode_D];
    time_t time_seed = time(NULL);
    for (int i = 0; i < nmode_A; i++)
    {
        srand(time_seed + idx_A[i]);
        extents_A[i] = randi(1, 4);
    }
    for (int i = 0; i < nmode_B; i++)
    {
        srand(time_seed + idx_B[i]);
        extents_B[i] = randi(1, 4);
    }
    for (int i = 0; i < nmode_D; i++)
    {
        srand(time_seed + idx_D[i]);
        extents_D[i] = randi(1, 4);
    }    
    int64_t* extents_C = new int64_t[nmode_C];
    std::copy(extents_D, extents_D + nmode_D, extents_C);
    
    int64_t* strides_A = calculate_simple_strides(nmode_A, extents_A);
    int64_t* strides_B = calculate_simple_strides(nmode_B, extents_B);
    int64_t* strides_C = calculate_simple_strides(nmode_C, extents_C);
    int64_t* strides_D = calculate_simple_strides(nmode_D, extents_D);

    int size_A = calculate_size(nmode_A, extents_A);
    int size_B = calculate_size(nmode_B, extents_B);
    int size_C = calculate_size(nmode_C, extents_C);
    int size_D = calculate_size(nmode_D, extents_D);
    
    float* data_A = create_tensor_data_s(size_A);
    float* data_B = create_tensor_data_s(size_B);
    float* data_C = create_tensor_data_s(size_C);
    float* data_D = create_tensor_data_s(size_D);
    
    float* data_E = copy_tensor_data_s(size_D, data_D);

    float alpha = rand_s();
    float beta = rand_s();

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

    TAPP_execute_product(plan, exec, &status, (void*)&alpha, (void*)data_A, (void*)data_B, (void*)&beta, (void*)data_C, (void*)data_D);

    run_tblis_mult_s(nmode_A, extents_A, strides_A, data_A, 0, idx_A,
                   nmode_B, extents_B, strides_B, data_B, 0, idx_B,
                   nmode_C, extents_C, strides_C, data_C, 0, idx_D,
                   nmode_D, extents_D, strides_D, data_E, 0, idx_D,
                   alpha, beta);

    bool result = compare_tensors_s(data_D, data_E, size_D);

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
          size_A, size_B, size_C, size_D] = generate_contraction_s();

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
    create_handle(&handle);
    TAPP_create_tensor_product(&plan, handle, 0, info_A, idx_A, 0, info_B, idx_B, 0, info_C, idx_C, 0, info_D, idx_D, TAPP_DEFAULT_PREC);
    TAPP_status status;

    TAPP_executor exec;
    create_executor(&exec);

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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(1, 4));
    
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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(1, 4));

    int64_t max_idx = 0;
    for (size_t i = 0; i < nmode_C; i++)
    {
        if (max_idx < idx_C[i])
        {
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
        if (nmode_C > 1)
        {
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
          size_A, size_B, size_C, size_D] = generate_contraction_s(-1, -1, randi(2, 4), randi(0, 4), 2);

    int scewed_index = randi(1, nmode_D - 1);
    int signs[2] = {-1, 1};
    strides_D[scewed_index] = random_choice(2, signs) * (strides_D[scewed_index - 1] * extents_D[scewed_index - 1] - randi(1, strides_D[scewed_index - 1] * extents_D[scewed_index - 1] - 1));

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
