/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2024
 */

#include "tapp_ex_imp.h"
#include "helpers.h"
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

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

int main(int argc, char const *argv[])
{
    printf("Contraction: \n");
    contraction();
    printf("Hadamard: \n");
    hadamard();
    printf("Complex: \n");
    complex_num();
    printf("Conjugate: \n");
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
    printf("Negative str: \n");
    negative_str();
    printf("Subtensors: \n");
    subtensors();
    TAPP_finalize();
    exit(0);
    return 0;
}

void contraction()
{
    int nmode_A = 3;
    int64_t extents_A[3] = {4, 3, 3};
    int64_t strides_A[3] = {1, 4, 12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    // int exec_id = 1;
    // exec = (intptr_t)&exec_id;
    TAPP_status status;

    double alpha = 1;

    double A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    double B[36] = {
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

    double beta = 0;

    double C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    double D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    TAPP_error error = TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    int message_len = TAPP_explain_error(error, 0, NULL);
    char *message_buff = malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    printf(message_buff);
    free(message_buff);

    print_tensor_d(nmode_D, extents_D, strides_D, D);

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
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {4, 4};
    int64_t strides_B[2] = {1, 4};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {4, 4};
    int64_t strides_C[2] = {1, 4};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {4, 4};
    int64_t strides_D[2] = {1, 4};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double alpha = 3;

    double A[16] = {
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4};

    double B[16] = {
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4};

    double beta = 2;

    double C[16] = {
        1, 2, 1, 2,
        1, 2, 1, 2,
        1, 2, 1, 2,
        1, 2, 1, 2};

    double D[16] = {
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

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);

    print_tensor_d(nmode_D, extents_D, strides_D, D);

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
    TAPP_create_tensor_info(&info_A, TAPP_C64, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {3, 3};
    int64_t strides_B[2] = {1, 3};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C64, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C64, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 3};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double complex alpha = 1;

    double complex A[9] = {
        1 + 1 * I, 3 + 2 * I, 5 + 3 * I,
        1 + 1 * I, 3 + 2 * I, 5 + 3 * I,
        1 + 1 * I, 3 + 2 * I, 5 + 3 * I};

    double complex B[9] = {
        1 + 1 * I, 1 + 1 * I, 1 + 1 * I,
        2 + 2 * I, 2 + 2 * I, 2 + 2 * I,
        3 + 3 * I, 3 + 3 * I, 3 + 3 * I};

    double complex beta = 1 * I;

    double complex C[9] = {
        1 + 2 * I, 2 + 1 * I, 3 + 1 * I,
        1 + 2 * I, 2 + 1 * I, 3 + 1 * I,
        1 + 2 * I, 2 + 1 * I, 3 + 1 * I};

    double complex D[9] = {
        1 + 1 * I, 2 + 2 * I, 3 + 3 * I,
        4 + 4 * I, 5 + 5 * I, 6 + 6 * I,
        7 + 7 * I, 8 + 8 * I, 9 + 2 * I};

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);

    print_tensor_z(nmode_D, extents_D, strides_D, D);

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
    TAPP_create_tensor_info(&info_A, TAPP_C64, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {3, 3};
    int64_t strides_B[2] = {1, 3};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C64, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C64, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 3};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double complex alpha = 1;

    double complex A[9] = {
        1 + 1 * I, 3 + 2 * I, 5 + 3 * I,
        1 + 1 * I, 3 + 2 * I, 5 + 3 * I,
        1 + 1 * I, 3 + 2 * I, 5 + 3 * I};

    double complex B[9] = {
        1 + 1 * I, 1 + 1 * I, 1 + 1 * I,
        2 + 2 * I, 2 + 2 * I, 2 + 2 * I,
        3 + 3 * I, 3 + 3 * I, 3 + 3 * I};

    double complex beta = 1 * I;

    double complex C[9] = {
        1 + 2 * I, 2 + 1 * I, 3 + 1 * I,
        1 + 2 * I, 2 + 1 * I, 3 + 1 * I,
        1 + 2 * I, 2 + 1 * I, 3 + 1 * I};

    double complex D[9] = {
        1 + 1 * I, 2 + 2 * I, 3 + 3 * I,
        4 + 4 * I, 5 + 5 * I, 6 + 6 * I,
        7 + 7 * I, 8 + 8 * I, 9 + 2 * I};

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);

    print_tensor_z(nmode_D, extents_D, strides_D, D);

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
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {3, 3};
    int64_t strides_B[2] = {1, 3};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 3};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double alpha = 1;

    double A[1] = {
        5};

    double B[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9};

    double beta = 0;

    double C[9] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1};

    double D[9] = {
        2, 2, 2,
        2, 2, 2,
        2, 2, 2};

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);

    print_tensor_d(nmode_D, extents_D, strides_D, D);

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
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 5;
    int64_t extents_B[5] = {3, 2, 1, 2, 3};
    int64_t strides_B[5] = {1, 3, 6, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double alpha = 1;

    double A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    double B[36] = {
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

    double beta = 0;

    double C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    double D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);

    print_tensor_d(nmode_D, extents_D, strides_D, D);

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
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 4;
    int64_t extents_C[4] = {4, 1, 2, 2};
    int64_t strides_C[4] = {1, 4, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 4;
    int64_t extents_D[4] = {4, 1, 2, 2};
    int64_t strides_D[4] = {1, 4, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double alpha = 1;

    double A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    double B[36] = {
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

    double beta = 0;

    double C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    double D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);

    print_tensor_d(nmode_D, extents_D, strides_D, D);

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
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double alpha = 2;

    double A[36] = {
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    double B[36] = {
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

    double beta = 0;

    double C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    double D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);

    printf("\tOperation 1:\n");
    print_tensor_d(nmode_D, extents_D, strides_D, D);

    alpha = 0.5;

    int nmode_E = 3;
    int64_t extents_E[3] = {4, 2, 2};
    int64_t strides_E[3] = {1, 4, 8};
    TAPP_tensor_info info_E;
    TAPP_create_tensor_info(&info_E, TAPP_F64, nmode_E, extents_E, strides_E);

    TAPP_tensor_product plan2;
    TAPP_element_op op_E = TAPP_IDENTITY;
    int64_t idx_E[3] = {'a', 'd', 'e'};
    TAPP_create_tensor_product(&plan2, handle, op_D, info_D, idx_D, op_C, info_C, idx_C, op_C, info_C, idx_C, op_E, info_E, idx_E, prec);

    double E[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};
    TAPP_execute_product(plan2, exec, &status, (void *)&alpha, (void *)D, (void *)C, (void *)&beta, (void *)C, (void *)E);

    printf("\tOperation 2:\n");
    print_tensor_d(nmode_E, extents_E, strides_E, E);

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
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {4, 4};
    int64_t strides_B[2] = {1, 4};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {4, 4};
    int64_t strides_C[2] = {1, 4};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {4, 4};
    int64_t strides_D[2] = {1, 4};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double alpha = 3;

    double A[16] = {
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4,
        1, 2, 3, 4};

    double B[16] = {
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4};

    double beta = 2;

    double C[16] = {
        1, 2, 1, 2,
        1, 2, 1, 2,
        1, 2, 1, 2,
        1, 2, 1, 2};

    double D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16};

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D);

    printf("\tOperation 1:\n");
    print_tensor_d(nmode_D, extents_D, strides_D, D);

    alpha = 1;
    beta = 2;
    double E[16] = {
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
    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A, (void *)D, (void *)&beta, (void *)C, (void *)E);

    printf("\tOperation 2:\n");
    print_tensor_d(nmode_D, extents_D, strides_D, E);

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void negative_str()
{
    int nmode_A = 3;
    int64_t extents_A[3] = {4, 3, 3};
    int64_t strides_A[3] = {-1, -4, -12};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {-1, -3, -6, -12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double alpha = 1;

    double A[36] = {
        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,

        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,

        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1,
        -1, 1.01, 2, 1};

    double B[36] = {
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

    double beta = 0;

    double C[16] = {
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    double D[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    double *A_ptr = &A[35];
    double *B_ptr = &B[35];

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_ptr, (void *)B_ptr, (void *)&beta, (void *)C, (void *)D);

    print_tensor_d(nmode_D, extents_D, strides_D, D);

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}

void subtensors()
{
    int nmode_A = 3;
    int64_t extents_A[3] = {3, 2, 2};
    int64_t strides_A[3] = {1, 12, 24};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A);

    int nmode_B = 3;
    int64_t extents_B[3] = {2, 2, 3};
    int64_t strides_B[3] = {3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    int nmode_C = 2;
    int64_t extents_C[2] = {3, 3};
    int64_t strides_C[2] = {1, 3};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    int nmode_D = 2;
    int64_t extents_D[2] = {3, 3};
    int64_t strides_D[2] = {1, 4};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    TAPP_handle handle;
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
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    create_executor(&exec);
    TAPP_status status;

    double alpha = 1;

    double A[48] = {
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

    double B[36] = {
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

    double beta = 0.5;

    double C[9] = {
        2, 4, 6,
        2, 4, 6,
        2, 4, 6};

    double D[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12};

    double *A_ptr = &A[5];

    double *B_ptr = &B[1];

    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_ptr, (void *)B_ptr, (void *)&beta, (void *)C, (void *)D);

    int64_t super_extents_D[2] = {4, 3};
    int64_t super_strides_D[2] = {1, 4};
    print_tensor_d(nmode_D, super_extents_D, super_strides_D, D);

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
}
