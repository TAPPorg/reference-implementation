/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#include "tblis_bind.h"

namespace {

void run_tblis_mult_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, float* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, float* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, float* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, float* D, int op_D, int64_t* idx_D,
                    float alpha, float beta);
bool compare_tensors_s(float* A, float* B, int size);
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, float*> contract_unique_idx_s(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2);

void run_tblis_mult_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, double* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, double* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, double* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, double* D, int op_D, int64_t* idx_D,
                    double alpha, double beta);
bool compare_tensors_d(double* A, double* B, int size);
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, double*> contract_unique_idx_d(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2);

void run_tblis_mult_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<float>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<float>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<float>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<float>* D, int op_D, int64_t* idx_D,
                    std::complex<float> alpha, std::complex<float> beta);
bool compare_tensors_c(std::complex<float>* A, std::complex<float>* B, int size);
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, std::complex<float>*> contract_unique_idx_c(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2);

void run_tblis_mult_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<double>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<double>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<double>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<double>* D, int op_D, int64_t* idx_D,
                    std::complex<double> alpha, std::complex<double> beta);
bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size);
std::tuple<tblis::tblis_tensor*, tblis::label_type*, tblis::len_type*, tblis::stride_type*, std::complex<double>*> contract_unique_idx_z(tblis::tblis_tensor* tensor, tblis::label_type* idx, int nmode_1, tblis::label_type* idx_1, int nmode_2, tblis::label_type* idx_2);



std::string str(bool b);
tblis::len_type* translate_extents_to_tblis(int nmode, int64_t* extents);
tblis::stride_type* translate_strides_to_tblis(int nmode, int64_t* strides);
tblis::label_type* translate_idx_to_tblis(int nmode, int64_t* idx);
void execute_product_tblis_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta);
void execute_product_tblis_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta);
void execute_product_tblis_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta);
void execute_product_tblis_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta);


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

    tblis::tblis_tensor_mult(tblis_single, NULL, tblis_A_reduced, tblis_idx_A_reduced, tblis_B_reduced, tblis_idx_B_reduced, &tblis_D, tblis_idx_D);

    tblis::tblis_tensor_add(tblis_single, NULL, &tblis_C, tblis_idx_C, &tblis_D, tblis_idx_D);

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

bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size_)
{
    bool found = false;
    for (int i = 0; i < size_; i++)
    {
        double rel_diff_r = abs((A[i].real() - B[i].real()) / (A[i].real() > B[i].real() ? A[i].real() : B[i].real()));
        double rel_diff_i = abs((A[i].imag() - B[i].imag()) / (A[i].imag() > B[i].imag() ? A[i].imag() : B[i].imag()));
        double abs_diff_r = abs(A[i].real() - B[i].real());
        double abs_diff_i = abs(A[i].imag() - B[i].imag());
        if ((rel_diff_r > 0.00005 || rel_diff_i > 0.00005) && (abs_diff_r > 1e-12 || abs_diff_i >  1e-12))
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << std::complex<double>(rel_diff_r, rel_diff_i) << std::endl;
            std::cout << "\n" << " size: " << size_ << ". " << std::endl;
            found = true;
        }
    }
    return !found;
}



std::string str(bool b)
{
    return b ? "true" : "false";
}

void execute_product_tblis_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta)
{
  float* A_ = static_cast<float*>(A);
  float* B_ = static_cast<float*>(B);
  float* C_ = static_cast<float*>(C);
  float* D_ = static_cast<float*>(D);
  float* alpha_ = static_cast<float*>(alpha);
  float* beta_ = static_cast<float*>(beta);

  run_tblis_mult_s(nmode_A, extents_A, strides_A, A_, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B_, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C_, op_C, idx_C,
                   nmode_D, extents_D, strides_D, D_, op_D, idx_D,
                   *alpha_, *beta_);
}

void execute_product_tblis_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta)
{
  double* A_ = static_cast<double*>(A);
  double* B_ = static_cast<double*>(B);
  double* C_ = static_cast<double*>(C);
  double* D_ = static_cast<double*>(D);
  double* alpha_ = static_cast<double*>(alpha);
  double* beta_ = static_cast<double*>(beta);

  run_tblis_mult_d(nmode_A, extents_A, strides_A, A_, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B_, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C_, op_C, idx_C,
                   nmode_D, extents_D, strides_D, D_, op_D, idx_D,
                   *alpha_, *beta_);
}

void execute_product_tblis_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta)
{
  std::complex<float>* A_ = static_cast<std::complex<float>*>(A);
  std::complex<float>* B_ = static_cast<std::complex<float>*>(B);
  std::complex<float>* C_ = static_cast<std::complex<float>*>(C);
  std::complex<float>* D_ = static_cast<std::complex<float>*>(D);
  std::complex<float>* alpha_ = static_cast<std::complex<float>*>(alpha);
  std::complex<float>* beta_ = static_cast<std::complex<float>*>(beta);

  run_tblis_mult_c(nmode_A, extents_A, strides_A, A_, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B_, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C_, op_C, idx_C,
                   nmode_D, extents_D, strides_D, D_, op_D, idx_D,
                   *alpha_, *beta_);
}

void execute_product_tblis_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta)
{
  std::complex<double>* A_ = static_cast<std::complex<double>*>(A);
  std::complex<double>* B_ = static_cast<std::complex<double>*>(B);
  std::complex<double>* C_ = static_cast<std::complex<double>*>(C);
  std::complex<double>* D_ = static_cast<std::complex<double>*>(D);
  std::complex<double>* alpha_ = static_cast<std::complex<double>*>(alpha);
  std::complex<double>* beta_ = static_cast<std::complex<double>*>(beta);

  run_tblis_mult_z(nmode_A, extents_A, strides_A, A_, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B_, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C_, op_C, idx_C,
                   nmode_D, extents_D, strides_D, D_, op_D, idx_D,
                   *alpha_, *beta_);
}


}

extern "C" {
  void bind_tblis_execute_product(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                    void* alpha, void* beta, int datatype_tapp){
    switch (datatype_tapp) {
      case TAPP_F32:
        execute_product_tblis_s(nmode_A, extents_A, strides_A, A, op_A, idx_A, nmode_B, extents_B, strides_B, B, op_B, idx_B,
                         nmode_C, extents_C, strides_C, C, op_C, idx_C, nmode_D, extents_D, strides_D, D, op_D, idx_D,
                         alpha, beta);
        break;
      case TAPP_F64:
        execute_product_tblis_d(nmode_A, extents_A, strides_A, A, op_A, idx_A, nmode_B, extents_B, strides_B, B, op_B, idx_B,
                         nmode_C, extents_C, strides_C, C, op_C, idx_C, nmode_D, extents_D, strides_D, D, op_D, idx_D,
                         alpha, beta);
        break;
      case TAPP_C32:
        execute_product_tblis_c(nmode_A, extents_A, strides_A, A, op_A, idx_A, nmode_B, extents_B, strides_B, B, op_B, idx_B,
                         nmode_C, extents_C, strides_C, C, op_C, idx_C, nmode_D, extents_D, strides_D, D, op_D, idx_D,
                         alpha, beta);
        break;
      case TAPP_C64:
        execute_product_tblis_z(nmode_A, extents_A, strides_A, A, op_A, idx_A, nmode_B, extents_B, strides_B, B, op_B, idx_B,
                         nmode_C, extents_C, strides_C, C, op_C, idx_C, nmode_D, extents_D, strides_D, D, op_D, idx_D,
                         alpha, beta);
        break;
    } 
  }

int compare_tensors_(void* A, void* B, int64_t size, int datatype_tapp){
    bool result = false;
    switch (datatype_tapp) { // tapp_datatype
      case TAPP_F32:
        {
          float* A_ = static_cast<float*>(A);
          float* B_ = static_cast<float*>(B);
          result = compare_tensors_s(A_, B_, (int)size);
        }
        break;
      case TAPP_F64:
        {
          double* A_ = static_cast<double*>(A);
          double* B_ = static_cast<double*>(B);
          result = compare_tensors_d(A_, B_, (int)size);
        }
        break;
      case TAPP_C32:
        {
          std::complex<float>* A_ = static_cast<std::complex<float>*>(A);
          std::complex<float>* B_ = static_cast<std::complex<float>*>(B);
          result = compare_tensors_c(A_, B_, (int)size);
        }
        break;
      case TAPP_C64:
        {
          std::complex<double>* A_ = static_cast<std::complex<double>*>(A);
          std::complex<double>* B_ = static_cast<std::complex<double>*>(B);
          result = compare_tensors_z(A_, B_, (int)size);
        }
        break;
    } 
    return result; 
  }
}

