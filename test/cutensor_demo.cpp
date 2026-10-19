/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - December 2025
 */

#include <tapp.h>
#include <tapp_cutensor.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <complex>
#include <cassert>

#include "helpers.h"

void contraction();
void hadamard();
void complex_num();
void conjugate();
void zero_dim();
void one_ext_contracted();
void one_ext_transfered();
void chained_diff_op();
void chained_same_op();
void negative_str();
void subtensors();
void permute();
void wait_one();
void attributes();
void multiple_devices();

int main(int argc, char const *argv[])
{
    printf("Contraction: \n");
    contraction();
    printf("Hadamard: \n");
    hadamard();
    printf("Complex: \n");
    complex_num();
    printf("Conjugate: \n"); // Incorrect result with CuTensor, CuTensor applies opC in another way
    conjugate();
    printf("Zero dim: \n");
    zero_dim();
    printf("One ext contracted: \n");
    one_ext_contracted();
    printf("One ext transfered: \n");
    one_ext_transfered();
    printf("Chained diff op: \n");
    chained_diff_op();
    printf("Chained same op: \n");
    chained_same_op();
    /*printf("Negative str: \n"); // Doesn't work with CuTensor, does not allow negative strides
    negative_str();*/
    printf("Subtensors: \n");
    subtensors();
    printf("Permute: \n");
    permute();
    printf("Wait one: \n");
    wait_one();
    printf("Attributes: \n");
    attributes();
    printf("Multiple devices: \n");
    multiple_devices();
    return 0;
}

void contraction()
{
    int nmode_A = 3;
    int64_t extents_A[3] = {4, 3, 3};
    int64_t strides_A[3] = {1, 4, 12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[3] = {'a', 'b', 'c'};
    int64_t idx_B[4] = {'c', 'd', 'e', 'b'};
    int64_t idx_C[3] = {'a', 'd', 'e'};
    int64_t idx_D[3] = {'a', 'd', 'e'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 1;

    float A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    float B[36] = {
        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6};

    float beta = 0;

    float C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    float D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 36 * sizeof(float));
    cudaMalloc((void**)&B_d, 36 * sizeof(float));
    cudaMalloc((void**)&C_d, 16 * sizeof(float));
    cudaMalloc((void**)&D_d, 16 * sizeof(float));

    cudaMemcpy(A_d, (void*)A, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);
   
    TAPP_error error = TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    int message_len = TAPP_explain_error(error, 0, NULL);
    char *message_buff = (char*)malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    printf("%s", message_buff);
    free(message_buff);

    TAPP_wait_all(exec);
    cudaMemcpy((void*)D, (void*)D_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    print_tensor_s(nmode_D, extents_D, strides_D, D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void hadamard()
{
    int nmode_A = 2;
    int64_t extents_A[2] = {4, 4};
    int64_t strides_A[2] = {1, 4};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {4, 4};
    int64_t strides_B[2] = {1, 4};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {4, 4};
    int64_t strides_C[2] = {1, 4};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {4, 4};
    int64_t strides_D[2] = {1, 4};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[2] = {'a', 'b'};
    int64_t idx_B[2] = {'a', 'b'};
    int64_t idx_C[2] = {'a', 'b'};
    int64_t idx_D[2] = {'a', 'b'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 3;

    float A[16] = {
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4};

    float B[16] = {
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4};

    float beta = 2;

    float C[16] = {
        1, 2, 1, 2,
        1, 2, 1, 2,
        1, 2, 1, 2,
        1, 2, 1, 2};

    float D[16] = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    };

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 16 * sizeof(float));
    cudaMalloc((void**)&B_d, 16 * sizeof(float));
    cudaMalloc((void**)&C_d, 16 * sizeof(float));
    cudaMalloc((void**)&D_d, 16 * sizeof(float));

    cudaMemcpy(A_d, (void*)A, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)D, (void*)D_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    print_tensor_s(nmode_D, extents_D, strides_D, D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void complex_num()
{
    int nmode_A = 2;
    int64_t extents_A[2] = {3, 3};
    int64_t strides_A[2] = {1, 3};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {3, 3};
    int64_t strides_B[2] = {1, 3};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 3};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[2] = {'a', 'b'};
    int64_t idx_B[2] = {'b', 'c'};
    int64_t idx_C[2] = {'a', 'c'};
    int64_t idx_D[2] = {'a', 'c'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    std::complex<float> alpha = 1;

    std::complex<float> A[9] = {
        {1, 1}, {3, 2}, {5, 3},
        {1, 1}, {3, 2}, {5, 3},
        {1, 1}, {3, 2}, {5, 3}};

    std::complex<float> B[9] = {
        {1, 1}, {1, 1}, {1, 1},
        {2, 2}, {2, 2}, {2, 2},
        {3, 3}, {3, 3}, {3, 3}};

    std::complex<float> beta = {0, 1};

    std::complex<float> C[9] = {
        {1, 2}, {2, 1}, {3, 1},
        {1, 2}, {2, 1}, {3, 1},
        {1, 2}, {2, 1}, {3, 1}};

    std::complex<float> D[9] = {
        {1, 1}, {2, 2}, {3, 3},
        {4, 4}, {5, 5}, {6, 6},
        {7, 7}, {8, 8}, {9, 2}};

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 9 * sizeof(std::complex<float>));
    cudaMalloc((void**)&B_d, 9 * sizeof(std::complex<float>));
    cudaMalloc((void**)&C_d, 9 * sizeof(std::complex<float>));
    cudaMalloc((void**)&D_d, 9 * sizeof(std::complex<float>));

    cudaMemcpy(A_d, (void*)A, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)D, (void*)D_d, 9 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    print_tensor_c(nmode_D, extents_D, strides_D, D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void conjugate()
{
    int nmode_A = 2;
    int64_t extents_A[2] = {3, 3};
    int64_t strides_A[2] = {1, 3};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {3, 3};
    int64_t strides_B[2] = {1, 3};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 3};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_CONJUGATE;
    TAPP_element_op op_C = TAPP_CONJUGATE;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[2] = {'a', 'b'};
    int64_t idx_B[2] = {'b', 'c'};
    int64_t idx_C[2] = {'a', 'c'};
    int64_t idx_D[2] = {'a', 'c'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    std::complex<float> alpha = 1;

    std::complex<float> A[9] = {
        {1, 1}, {3, 2}, {5, 3},
        {1, 1}, {3, 2}, {5, 3},
        {1, 1}, {3, 2}, {5, 3}};

    std::complex<float> B[9] = {
        {1, 1}, {1, 1}, {1, 1},
        {2, 2}, {2, 2}, {2, 2},
        {3, 3}, {3, 3}, {3, 3}};

    std::complex<float> beta = {0, 1};

    std::complex<float> C[9] = {
        {1, 2}, {2, 1}, {3, 1},
        {1, 2}, {2, 1}, {3, 1},
        {1, 2}, {2, 1}, {3, 1}};

    std::complex<float> D[9] = {
        {1, 1}, {2, 2}, {3, 3},
        {4, 4}, {5, 5}, {6, 6},
        {7, 7}, {8, 8}, {9, 2}};
        
    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 9 * sizeof(std::complex<float>));
    cudaMalloc((void**)&B_d, 9 * sizeof(std::complex<float>));
    cudaMalloc((void**)&C_d, 9 * sizeof(std::complex<float>));
    cudaMalloc((void**)&D_d, 9 * sizeof(std::complex<float>));

    cudaMemcpy(A_d, (void*)A, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)D, (void*)D_d, 9 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    print_tensor_c(nmode_D, extents_D, strides_D, D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void zero_dim()
{
    int nmode_A = 0;
    int64_t extents_A[0] = {};
    int64_t strides_A[0] = {};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {3, 3};
    int64_t strides_B[2] = {1, 3};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 3};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[0] = {};
    int64_t idx_B[2] = {'a', 'b'};
    int64_t idx_C[2] = {'a', 'b'};
    int64_t idx_D[2] = {'a', 'b'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 1;

    float A[1] = {
        5};

    float B[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};

    float beta = 0;

    float C[9] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1};

    float D[9] = {
        2, 2, 2,
        2, 2, 2,
        2, 2, 2};
        
    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 1 * sizeof(float));
    cudaMalloc((void**)&B_d, 9 * sizeof(float));
    cudaMalloc((void**)&C_d, 9 * sizeof(float));
    cudaMalloc((void**)&D_d, 9 * sizeof(float));

    cudaMemcpy(A_d, (void*)A, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 9 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)D, (void*)D_d, 9 * sizeof(float), cudaMemcpyDeviceToHost);

    print_tensor_s(nmode_D, extents_D, strides_D, D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void one_ext_contracted()
{
    int nmode_A = 4;
    int64_t extents_A[4] = {4, 1, 3, 3};
    int64_t strides_A[4] = {1, 4, 4, 12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 5;
    int64_t extents_B[5] = {3, 2, 1, 2, 3};
    int64_t strides_B[5] = {1, 3, 6, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[4] = {'a', 'b', 'c', 'd'};
    int64_t idx_B[5] = {'d', 'e', 'b', 'f', 'c'};
    int64_t idx_C[3] = {'a', 'e', 'f'};
    int64_t idx_D[3] = {'a', 'e', 'f'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 1;

    float A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    float B[36] = {
        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6};

    float beta = 0;

    float C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    float D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 36 * sizeof(float));
    cudaMalloc((void**)&B_d, 36 * sizeof(float));
    cudaMalloc((void**)&C_d, 16 * sizeof(float));
    cudaMalloc((void**)&D_d, 16 * sizeof(float));

    cudaMemcpy(A_d, (void*)A, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)D, (void*)D_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    print_tensor_s(nmode_D, extents_D, strides_D, D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void one_ext_transfered()
{
    int nmode_A = 4;
    int64_t extents_A[4] = {4, 1, 3, 3};
    int64_t strides_A[4] = {1, 4, 4, 12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 4;
    int64_t extents_C[4] = {4, 1, 2, 2};
    int64_t strides_C[4] = {1, 4, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 4;
    int64_t extents_D[4] = {4, 1, 2, 2};
    int64_t strides_D[4] = {1, 4, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[4] = {'a', 'b', 'c', 'd'};
    int64_t idx_B[4] = {'d', 'e', 'f', 'c'};
    int64_t idx_C[4] = {'a', 'b', 'e', 'f'};
    int64_t idx_D[4] = {'a', 'b', 'e', 'f'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 1;

    float A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    float B[36] = {
        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6};

    float beta = 0;

    float C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    float D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 36 * sizeof(float));
    cudaMalloc((void**)&B_d, 36 * sizeof(float));
    cudaMalloc((void**)&C_d, 16 * sizeof(float));
    cudaMalloc((void**)&D_d, 16 * sizeof(float));

    cudaMemcpy(A_d, (void*)A, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)D, (void*)D_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    print_tensor_s(nmode_D, extents_D, strides_D, D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void chained_diff_op()
{
    int nmode_A = 3;
    int64_t extents_A[3] = {4, 3, 3};
    int64_t strides_A[3] = {1, 4, 12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[3] = {'a', 'b', 'c'};
    int64_t idx_B[4] = {'c', 'd', 'e', 'b'};
    int64_t idx_C[3] = {'a', 'd', 'e'};
    int64_t idx_D[3] = {'a', 'd', 'e'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 2;

    float A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    float B[36] = {
        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6};

    float beta = 0;

    float C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    float D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 36 * sizeof(float));
    cudaMalloc((void**)&B_d, 36 * sizeof(float));
    cudaMalloc((void**)&C_d, 16 * sizeof(float));
    cudaMalloc((void**)&D_d, 16 * sizeof(float));

    cudaMemcpy(A_d, (void*)A, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)D, (void*)D_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\tOperation 1:\n");
    print_tensor_s(nmode_D, extents_D, strides_D, D);

    alpha = 0.5;

    int nmode_E = 3;
    int64_t extents_E[3] = {4, 2, 2};
    int64_t strides_E[3] = {1, 4, 8};
    TAPP_tensor_info info_E;
    TAPP_create_tensor_info(&info_E, TAPP_F32, nmode_E, extents_E, strides_E);

    TAPP_tensor_product plan2;
    TAPP_element_op op_E = TAPP_IDENTITY;
    int64_t idx_E[3] = {'a', 'd', 'e'};
    TAPP_create_tensor_product(&plan2, exec, op_D, info_D, idx_D, op_C, info_C, idx_C, op_C, info_C, idx_C, op_E, info_E, idx_E, prec);

    float E[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    void* E_d; // Device pointer
    cudaMalloc((void**)&E_d, 16 * sizeof(float));

    cudaMemcpy(E_d, (void*)E, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(E_d) % 128 == 0);
    
    TAPP_execute_product(plan2, exec, &status, (void *)&alpha, (void *)D_d, (void *)C_d, (void *)&beta, (void *)C_d, (void *)E_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)E, (void*)E_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\tOperation 2:\n");
    print_tensor_s(nmode_E, extents_E, strides_E, E);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    if (E_d) cudaFree(E_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_product(plan2);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_tensor_info(info_E);
    TAPP_destroy_executor(exec);
}

void chained_same_op()
{
    int nmode_A = 2;
    int64_t extents_A[2] = {4, 4};
    int64_t strides_A[2] = {1, 4};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {4, 4};
    int64_t strides_B[2] = {1, 4};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {4, 4};
    int64_t strides_C[2] = {1, 4};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {4, 4};
    int64_t strides_D[2] = {1, 4};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[2] = {'a', 'b'};
    int64_t idx_B[2] = {'a', 'b'};
    int64_t idx_C[2] = {'a', 'b'};
    int64_t idx_D[2] = {'a', 'b'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 3;

    float A[16] = {
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4};

    float B[16] = {
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4};

    float beta = 2;

    float C[16] = {
        1, 2, 1, 2,
        1, 2, 1, 2,
        1, 2, 1, 2,
        1, 2, 1, 2};

    float D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16};

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 16 * sizeof(float));
    cudaMalloc((void**)&B_d, 16 * sizeof(float));
    cudaMalloc((void**)&C_d, 16 * sizeof(float));
    cudaMalloc((void**)&D_d, 16 * sizeof(float));

    cudaMemcpy(A_d, (void*)A, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)D, (void*)D_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\tOperation 1:\n");
    print_tensor_s(nmode_D, extents_D, strides_D, D);

    alpha = 1;
    beta = 2;
    float E[16] = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    };

    void* E_d; // Device pointer
    cudaMalloc((void**)&E_d, 16 * sizeof(float));

    cudaMemcpy(E_d, (void*)E, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(E_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)D_d, (void *)&beta, (void *)C_d, (void *)E_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)E, (void*)E_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\tOperation 2:\n");
    print_tensor_s(nmode_D, extents_D, strides_D, E);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

/*void negative_str() //cutensor does not support negative strides
{
    int nmode_A = 3;
    int64_t extents_A[3] = {4, 3, 3};
    int64_t strides_A[3] = {-1, -4, -12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, exec, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {-1, -3, -6, -12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, exec, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, exec, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, exec, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[3] = {'a', 'b', 'c'};
    int64_t idx_B[4] = {'c', 'd', 'e', 'b'};
    int64_t idx_C[3] = {'a', 'd', 'e'};
    int64_t idx_D[3] = {'a', 'd', 'e'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);
    TAPP_status status;

    float alpha = 1;

    float A[36] = {
        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,

        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,

        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1};

    float B[36] = {
        6, 6, 6,
        3, 3, 3,

        2, 2, 2,
        1, 1, 1,

        6, 6, 6,
        3, 3, 3,

        2, 2, 2,
        1, 1, 1,

        6, 6, 6,
        3, 3, 3,

        2, 2, 2,
        1, 1, 1};

    float beta = 0;

    float C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    float D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    float *A_ptr = &A[35];
    float *B_ptr = &B[35];

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_ptr, (void *)B_ptr, (void *)&beta, (void *)C, (void *)D);
    TAPP_wait_all(exec);

    print_tensor_s(nmode_D, extents_D, strides_D, D);

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
   
}*/

void subtensors()
{
    int nmode_A = 3;
    int64_t extents_A[3] = {3, 2, 2};
    int64_t strides_A[3] = {1, 12, 24};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 3;
    int64_t extents_B[3] = {2, 2, 3};
    int64_t strides_B[3] = {3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 3};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[3] = {'a', 'b', 'c'};
    int64_t idx_B[3] = {'b', 'c', 'd'};
    int64_t idx_C[2] = {'a', 'd'};
    int64_t idx_D[2] = {'a', 'd'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 1;

    float A[48] = {
        0,
        0,
        0,
        0,
        0,
        2,
        1.01,
        -1,
        0,
        0,
        0,
        0,

        0,
        0,
        0,
        0,
        0,
        2,
        1.01,
        -1,
        0,
        0,
        0,
        0,

        0,
        0,
        0,
        0,
        0,
        2,
        1.01,
        -1,
        0,
        0,
        0,
        0,

        0,
        0,
        0,
        0,
        0,
        2,
        1.01,
        -1,
        0,
        0,
        0,
        0,
    };

    float B[36] = {
        0, 1, 0,
        0, 2, 0,

        0, 3, 0,
        0, 4, 0,

        0, 2, 0,
        0, 4, 0,

        0, 6, 0,
        0, 8, 0,

        0, 3, 0,
        0, 6, 0,

        0, 9, 0,
        0, 12, 0};

    float beta = 0.5;

    float C[9] = {
        2, 4, 6,
        2, 4, 6,
        2, 4, 6};

    float D[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12};

    float *A_ptr = &A[5];

    float *B_ptr = &B[1];

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 43 * sizeof(float));
    cudaMalloc((void**)&B_d, 35 * sizeof(float));
    cudaMalloc((void**)&C_d, 9 * sizeof(float));
    cudaMalloc((void**)&D_d, 12 * sizeof(float));

    cudaMemcpy(A_d, (void*)A_ptr, 43 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B_ptr, 35 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(D_d, (void*)D, 12 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_all(exec);

    int64_t super_extents_D[2] = {4, 3};
    int64_t super_strides_D[2] = {1, 4};

    cudaMemcpy((void*)D, (void*)D_d, 12 * sizeof(float), cudaMemcpyDeviceToHost);
    print_tensor_s(nmode_D, super_extents_D, super_strides_D, D);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void permute(){
    int nmode_A = 2;
    int64_t extents_A[2] = {3, 3};
    int64_t strides_A[2] = {1, 3};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {3, 3};
    int64_t strides_B[2] = {1, 3};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, nmode_C, extents_C, strides_C);

    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_permute plan;
    TAPP_element_op op_A = TAPP_CONJUGATE;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_CONJUGATE;
    int64_t idx_A[2] = {'a', 'b'};
    int64_t idx_B[2] = {'a', 'b'};
    int64_t idx_C[2] = {'a', 'b'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_permute(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, prec);

    TAPP_status status;

    std::complex<float> alpha = 1;

    std::complex<float> A[9] = {
        {1, 1}, {3, 2}, {5, 3},
        {1, 1}, {3, 2}, {5, 3},
        {1, 1}, {3, 2}, {5, 3}};

    std::complex<float> B[9] = {
        {1, 1}, {1, 1}, {1, 1},
        {2, 2}, {2, 2}, {2, 2},
        {3, 3}, {3, 3}, {3, 3}};

    std::complex<float> beta = {0, 1};

    std::complex<float> C[9] = {
        {1, 2}, {2, 1}, {3, 1},
        {1, 2}, {2, 1}, {3, 1},
        {1, 2}, {2, 1}, {3, 1}};

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 9 * sizeof(std::complex<float>));
    cudaMalloc((void**)&B_d, 9 * sizeof(std::complex<float>));
    cudaMalloc((void**)&C_d, 9 * sizeof(std::complex<float>));

    cudaMemcpy(A_d, (void*)A, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 9 * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);

    TAPP_execute_permute(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d);
    TAPP_wait_all(exec);

    cudaMemcpy((void*)C, (void*)C_d, 9 * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    print_tensor_c(nmode_C, extents_C, strides_C, C);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    TAPP_destroy_tensor_permute(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_executor(exec);
}

// Copy of chained_diff_op, but replacing wail_all by wail_one
void wait_one()
{
    int nmode_A = 3;
    int64_t extents_A[3] = {4, 3, 3};
    int64_t strides_A[3] = {1, 4, 12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);


    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[3] = {'a', 'b', 'c'};
    int64_t idx_B[4] = {'c', 'd', 'e', 'b'};
    int64_t idx_C[3] = {'a', 'd', 'e'};
    int64_t idx_D[3] = {'a', 'd', 'e'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_status status;

    float alpha = 2;

    float A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    float B[36] = {
        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6};

    float beta = 0;

    float C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    float D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, 36 * sizeof(float));
    cudaMalloc((void**)&B_d, 36 * sizeof(float));
    cudaMalloc((void**)&C_d, 16 * sizeof(float));
    cudaMalloc((void**)&D_d, 16 * sizeof(float));

    cudaMemcpy(A_d, (void*)A, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, (void*)B, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d) % 128 == 0);
    assert(uintptr_t(B_d) % 128 == 0);
    assert(uintptr_t(C_d) % 128 == 0);
    assert(uintptr_t(D_d) % 128 == 0);

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    TAPP_wait_one(exec, status);

    cudaMemcpy((void*)D, (void*)D_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\tOperation 1:\n");
    print_tensor_s(nmode_D, extents_D, strides_D, D);

    alpha = 0.5;

    int nmode_E = 3;
    int64_t extents_E[3] = {4, 2, 2};
    int64_t strides_E[3] = {1, 4, 8};
    TAPP_tensor_info info_E;
    TAPP_create_tensor_info(&info_E, TAPP_F32, nmode_E, extents_E, strides_E);

    TAPP_tensor_product plan2;
    TAPP_element_op op_E = TAPP_IDENTITY;
    int64_t idx_E[3] = {'a', 'd', 'e'};
    TAPP_create_tensor_product(&plan2, exec, op_D, info_D, idx_D, op_C, info_C, idx_C, op_C, info_C, idx_C, op_E, info_E, idx_E, prec);

    float E[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    void* E_d; // Device pointer
    cudaMalloc((void**)&E_d, 16 * sizeof(float));

    cudaMemcpy(E_d, (void*)E, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(E_d) % 128 == 0);
    
    TAPP_execute_product(plan2, exec, &status, (void *)&alpha, (void *)D_d, (void *)C_d, (void *)&beta, (void *)C_d, (void *)E_d);
    TAPP_wait_one(exec, status);

    cudaMemcpy((void*)E, (void*)E_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\tOperation 2:\n");
    print_tensor_s(nmode_E, extents_E, strides_E, E);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    if (E_d) cudaFree(E_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_product(plan2);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_tensor_info(info_E);
    TAPP_destroy_executor(exec);
}

void attributes(){
    TAPP_attr attr;
    TAPP_create_attr(&attr);

    int* device_id_default;
    size_t size;
    cudaStream_t* stream_default;

    printf("---------------------------\n");
    printf("Default values of attributes:\n");
    printf("---------------------------\n");
    TAPP_attr_get(attr, TAPP_CUTENSOR_DEVICE, &size, (void**) &device_id_default);
    printf("TAPP_CUTENSOR_DEVICE==nullptr(should be 1): %i\n", device_id_default==nullptr);
    printf("TAPP_CUTENSOR_DEVICE size (should be %i): %i\n", 0, size);
    TAPP_attr_get(attr, TAPP_CUTENSOR_STREAM, &size, (void**) &stream_default);
    printf("Is TAPP_CUTENSOR_STREAM nullptr (should be 1): %i\n", stream_default==nullptr);
    printf("TAPP_CUTENSOR_STREAM size (should be %i): %i\n", 0, size);

    printf("---------------------------\n");
    printf("Values of attributes after update:\n");
    printf("---------------------------\n");
    int device_id = 1;
    void* value = &device_id;
    TAPP_attr_set(attr, TAPP_CUTENSOR_DEVICE, sizeof(int), (void*) &device_id);
    TAPP_attr_get(attr, TAPP_CUTENSOR_DEVICE, &size,  &value);
    printf("TAPP_CUTENSOR_DEVICE (should be 1): %i\n", *(int*)value);
    printf("TAPP_CUTENSOR_DEVICE size (should be %i): %i\n", sizeof(int), size);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    TAPP_attr_set(attr, TAPP_CUTENSOR_STREAM, sizeof(cudaStream_t), (void*) &stream);
    value = &stream;
    TAPP_attr_get(attr, TAPP_CUTENSOR_STREAM, &size, &value);
    printf("Is TAPP_CUTENSOR_STREAM nullptr (should be 0): %i\n", value==nullptr);
    printf("TAPP_CUTENSOR_STREAM size (should be %i): %i\n\n", sizeof(cudaStream_t), size);

    TAPP_destroy_attr(attr);
}

// Launch 2 contractions on 2 devices
void multiple_devices(){
    int nmode_A = 3;
    int64_t extents_A[3] = {4, 3, 3};
    int64_t strides_A[3] = {1, 4, 12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    int64_t idx_A[3] = {'a', 'b', 'c'};
    int64_t idx_B[4] = {'c', 'd', 'e', 'b'};
    int64_t idx_C[3] = {'a', 'd', 'e'};
    int64_t idx_D[3] = {'a', 'd', 'e'};
    TAPP_prectype prec = TAPP_DEFAULT_PREC;

    float alpha = 1;

    float A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    float B[36] = {
        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6,

        1, 1, 1,
        2, 2, 2,

        3, 3, 3,
        6, 6, 6};

    float beta = 0;

    float C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    float D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    // Device 1
    cudaSetDevice(1);
    TAPP_attr attr1;
    TAPP_create_attr(&attr1);
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    int device_id;
    device_id = 1;
    TAPP_attr_set(attr1, TAPP_CUTENSOR_DEVICE, sizeof(int), (void*) &device_id);
    TAPP_attr_set(attr1, TAPP_CUTENSOR_STREAM, sizeof(cudaStream_t), (void*) &stream1);

    TAPP_executor exec1;
    TAPP_create_executor(&exec1, attr1);
    TAPP_tensor_product plan1;
    TAPP_create_tensor_product(&plan1, exec1, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);
    void *A_d1, *B_d1, *C_d1, *D_d1; // Device pointers
    cudaMalloc((void**)&A_d1, 36 * sizeof(float));
    cudaMalloc((void**)&B_d1, 36 * sizeof(float));
    cudaMalloc((void**)&C_d1, 16 * sizeof(float));
    cudaMalloc((void**)&D_d1, 16 * sizeof(float));

    cudaMemcpy(A_d1, (void*)A, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d1, (void*)B, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d1, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d1) % 128 == 0);
    assert(uintptr_t(B_d1) % 128 == 0);
    assert(uintptr_t(C_d1) % 128 == 0);
    assert(uintptr_t(D_d1) % 128 == 0);
   
    TAPP_status status1;
    TAPP_error error = TAPP_execute_product(plan1, exec1, &status1, (void *)&alpha, (void *)A_d1, (void *)B_d1, (void *)&beta, (void *)C_d1, (void *)D_d1);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    int message_len = TAPP_explain_error(error, 0, NULL);
    char *message_buff = (char*)malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    printf("%s", message_buff);
    free(message_buff);
    TAPP_wait_all(exec1);

    // Device 2
    cudaSetDevice(2);
    TAPP_attr attr2;
    TAPP_create_attr(&attr2);
    cudaStream_t stream2;
    cudaStreamCreate(&stream2);
    device_id = 2;
    TAPP_attr_set(attr2, TAPP_CUTENSOR_DEVICE, sizeof(int), (void*) &device_id);
    TAPP_attr_set(attr2, TAPP_CUTENSOR_STREAM, sizeof(cudaStream_t), (void*) &stream2);

    TAPP_executor exec2;
    TAPP_create_executor(&exec2, attr2);
    TAPP_tensor_product plan2;
    TAPP_create_tensor_product(&plan2, exec2, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);
    void *A_d2, *B_d2, *C_d2, *D_d2; // Device pointers
    cudaMalloc((void**)&A_d2, 36 * sizeof(float));
    cudaMalloc((void**)&B_d2, 36 * sizeof(float));
    cudaMalloc((void**)&C_d2, 16 * sizeof(float));
    cudaMalloc((void**)&D_d2, 16 * sizeof(float));

    cudaMemcpy(A_d2, (void*)A, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d2, (void*)B, 36 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d2, (void*)C, 16 * sizeof(float), cudaMemcpyHostToDevice);

    assert(uintptr_t(A_d2) % 128 == 0);
    assert(uintptr_t(B_d2) % 128 == 0);
    assert(uintptr_t(C_d2) % 128 == 0);
    assert(uintptr_t(D_d2) % 128 == 0);
   
    TAPP_status status2;
    error = TAPP_execute_product(plan1, exec2, &status2, (void *)&alpha, (void *)A_d2, (void *)B_d2, (void *)&beta, (void *)C_d2, (void *)D_d2);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*)malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    printf("%s", message_buff);
    free(message_buff);
    TAPP_wait_all(exec2);

    // Result 1
    cudaMemcpy((void*)D, (void*)D_d1, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    print_tensor_s(nmode_D, extents_D, strides_D, D);

    // Result 2
    cudaMemcpy((void*)D, (void*)D_d2, 16 * sizeof(float), cudaMemcpyDeviceToHost);
    print_tensor_s(nmode_D, extents_D, strides_D, D);
    
    // Cleanup
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);

    if (A_d1) cudaFree(A_d1);
    if (B_d1) cudaFree(B_d1);
    if (C_d1) cudaFree(C_d1);
    if (D_d1) cudaFree(D_d1);
    TAPP_destroy_tensor_product(plan1);
    TAPP_destroy_executor(exec1);
    TAPP_destroy_attr(attr1);

    if (A_d2) cudaFree(A_d2);
    if (B_d2) cudaFree(B_d2);
    if (C_d2) cudaFree(C_d2);
    if (D_d2) cudaFree(D_d2);
    TAPP_destroy_tensor_product(plan2);
    TAPP_destroy_executor(exec2);
    TAPP_destroy_attr(attr2);
}
