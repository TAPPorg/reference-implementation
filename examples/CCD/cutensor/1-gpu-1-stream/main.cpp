/*
 * Contractions are assumed to barely fit on a device, meaning 1 operation at a time per device
 * Generate input, intermediate and result tensors and save their values for CCD benchmark
 * Result tensor: Z2 vvoo
 *
 * List of input tensors:
 * V1 oovv
 * T1 vvoo
 * F1 oo
 * V2 vvvv
 * V3 voov
 * F2 vv
 * V4 oooo
 *
 * List of intermediate tensors vvoo unless specified:
 * D1
 * G1
 * I1
 * J1 vv
 * M1
 * X1 ovvo
 * Y1
 * A2 oo
 * E2 oooo
 * G2
 */

#include <tapp.h>
#include "tapp_cutensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <omp.h>

#include <stdlib.h>
#include <stdio.h>
#include <filesystem> 

const int SEED = 123;
const double ALPHA_ZERO = 0;
const double ALPHA_ONE = 1;
const double ALPHA_MINUS_ONE = -1;
const double ALPHA_ONE_HALF = 1.0/2.0;
const double ALPHA_MINUS_ONE_HALF = - 1.0/2.0;
const double ALPHA_MINUS_ONE_QUARTER = - 1.0/4.0;
const double ALPHA_ONE_QUARTER = 1.0/4.0;
const double ALPHA_TWO = 2;
const double BETA_ZERO = 0;
const double BETA_ONE = 1;
const double BETA_MINUS_ONE = -1;
void strides4(const int64_t e[4], int64_t s[4]);
void strides3(const int64_t e[3], int64_t s[3]);
void strides2(const int64_t e[2], int64_t s[2]);
void save_tensor(double*, size_t, std::string);
void read_tensor(double*, size_t, std::string, int64_t, int64_t);
void set_tensor_to_zero(double* T, size_t n);

int main(int argc, char const *argv[])
{
    printf("Start...\n");
    const int64_t nocc = std::stoi(argv[1]);
    const int64_t nvir = std::stoi(argv[2]);
    const int threads = 1;

    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;
    TAPP_prectype prec = TAPP_DEFAULT_PREC;

    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    // Generate and save all input tensors
    printf("Reading tensors from TBLIS reference...\n");
    // V1 oovv
    int nmode_V1 = 4;
    int64_t extents_V1[4] = {nocc, nocc, nvir, nvir};
    int64_t strides_V1[4];
    strides4(extents_V1, strides_V1);
    TAPP_tensor_info info_V1;
    TAPP_create_tensor_info(&info_V1, TAPP_F64, nmode_V1, extents_V1, strides_V1);
    size_t num_V1 = nocc*nocc*nvir*nvir;
    double V1[num_V1];
    std::string name = "V1";
    read_tensor(V1, num_V1, name, nocc, nvir);
    void* V1_d = nullptr;
    cudaMalloc(&V1_d, sizeof(double)*num_V1);
    cudaMemcpy(V1_d, V1, num_V1 * sizeof(double), cudaMemcpyHostToDevice);
    printf("V1 done!\n");

    // T1 vvoo
    int nmode_T1 = 4;
    int64_t extents_T1[4] = {nvir, nvir, nocc, nocc};
    int64_t strides_T1[4];
    strides4(extents_T1, strides_T1);
    TAPP_tensor_info info_T1;
    TAPP_create_tensor_info(&info_T1, TAPP_F64, nmode_T1, extents_T1, strides_T1);
    size_t num_T1 = nocc*nocc*nvir*nvir;
    double T1[num_T1];
    name = "T1";
    read_tensor(T1, num_T1, name, nocc, nvir);
    void* T1_d = nullptr;
    cudaMalloc(&T1_d, sizeof(double)*num_T1);
    cudaMemcpy(T1_d, T1, num_T1 * sizeof(double), cudaMemcpyHostToDevice);
    printf("T1 done!\n");

    // F1 oo
    int nmode_F1 = 2;
    int64_t extents_F1[2] = {nocc, nocc};
    int64_t strides_F1[2];
    strides2(extents_F1, strides_F1);
    TAPP_tensor_info info_F1;
    TAPP_create_tensor_info(&info_F1, TAPP_F64, nmode_F1, extents_F1, strides_F1);
    size_t num_F1 = nocc*nocc;
    double F1[num_F1];
    void* F1_d = nullptr;
    name = "F1";
    read_tensor(F1, num_F1, name, nocc, nvir);
    printf("F1 done!\n");

    // V2 vvvv
    int nmode_V2 = 4;
    int64_t extents_V2[4] = {nvir, nvir, nvir, nvir};
    int64_t strides_V2[4];
    strides4(extents_V2, strides_V2);
    TAPP_tensor_info info_V2;
    TAPP_create_tensor_info(&info_V2, TAPP_F64, nmode_V2, extents_V2, strides_V2);
    size_t num_V2 = nvir*nvir*nvir*nvir;
    double V2[num_V2];
    void* V2_d = nullptr;
    name = "V2";
    read_tensor(V2, num_V2, name, nocc, nvir);
    printf("V2 done!\n");

    // V3 voov
    int nmode_V3 = 4;
    int64_t extents_V3[4] = {nvir, nocc, nocc, nvir};
    int64_t strides_V3[4];
    strides4(extents_V3, strides_V3);
    TAPP_tensor_info info_V3;
    TAPP_create_tensor_info(&info_V3, TAPP_F64, nmode_V3, extents_V3, strides_V3);
    size_t num_V3 = nvir*nocc*nocc*nvir;
    double V3[num_V3];
    void* V3_d = nullptr;
    name = "V3";
    read_tensor(V3, num_V3, name, nocc, nvir);
    printf("V3 done!\n");

    // F2 vv
    int nmode_F2 = 2;
    int64_t extents_F2[2] = {nvir, nvir};
    int64_t strides_F2[2];
    strides2(extents_F2, strides_F2);
    TAPP_tensor_info info_F2;
    TAPP_create_tensor_info(&info_F2, TAPP_F64, nmode_F2, extents_F2, strides_F2);
    size_t num_F2 = nvir*nvir;
    double F2[num_F2];
    void* F2_d = nullptr;
    name = "F2";
    read_tensor(F2, num_F2, name, nocc, nvir);
    printf("F2 done!\n");

    // V4 oooo
    int nmode_V4 = 4;
    int64_t extents_V4[4] = {nocc, nocc, nocc, nocc};
    int64_t strides_V4[4];
    strides4(extents_V4, strides_V4);
    TAPP_tensor_info info_V4;
    TAPP_create_tensor_info(&info_V4, TAPP_F64, nmode_V4, extents_V4, strides_V4);
    size_t num_V4 = nocc*nocc*nocc*nocc;
    double V4[num_V4];
    void* V4_d = nullptr;
    name = "V4";
    read_tensor(V4, num_V4, name, nocc, nvir);
    printf("V4 done!\n");

    // V5 vvoo
    int nmode_V5 = 4;
    int64_t extents_V5[4] = {nvir, nvir, nocc, nocc};
    int64_t strides_V5[4];
    strides4(extents_V5, strides_V5);
    TAPP_tensor_info info_V5;
    TAPP_create_tensor_info(&info_V5, TAPP_F64, nmode_V5, extents_V5, strides_V5);
    size_t num_V5 = nvir*nvir*nocc*nocc;
    double V5[num_V5];
    void* V5_d = nullptr;
    name = "V5";
    read_tensor(V5, num_V5, name, nocc, nvir);
    printf("V5 done!\n");

    // Z2 vvoo
    int nmode_Z2 = 4;
    int64_t extents_Z2[4] = {nvir, nvir, nocc, nocc};
    int64_t strides_Z2[4];
    strides4(extents_Z2, strides_Z2);
    TAPP_tensor_info info_Z2;
    TAPP_create_tensor_info(&info_Z2, TAPP_F64, nmode_Z2, extents_Z2, strides_Z2);
    size_t num_Z2 = nocc*nocc*nvir*nvir;
    double Z2[num_Z2];
    void* Z2_d = nullptr;
    cudaMalloc(&Z2_d, sizeof(double)*num_Z2);
    printf("Z2 done!\n");

    // Tmp vvoo
    int nmode_Tmp = 4;
    int64_t extents_Tmp[4] = {nvir, nvir, nocc, nocc};
    int64_t strides_Tmp[4];
    strides4(extents_Tmp, strides_Tmp);
    TAPP_tensor_info info_Tmp;
    TAPP_create_tensor_info(&info_Tmp, TAPP_F64, nmode_Tmp, extents_Tmp, strides_Tmp);
    size_t num_Tmp = nocc*nocc*nvir*nvir;
    void* Tmp_d = nullptr;
    cudaMalloc(&Tmp_d, sizeof(double)*num_Tmp);
    double Tmp[num_Tmp];

    // Compute intermediate -> Accumulate into result tensor -> Destroy intermediate
    //////////////////////////////////
    // Contractions with D1 intermediate: D1(f,e,m,n) = F1(l,m) T1(f,e,n,l)
    //////////////////////////////////

    cudaMalloc(&F1_d, sizeof(double)*num_F1);
    cudaMemcpy(F1_d, F1, num_F1 * sizeof(double), cudaMemcpyHostToDevice);

    TAPP_tensor_product plan_1;
    int64_t idx_F1_1[2] = {'l', 'm'};
    int64_t idx_T1_1[4] = {'f', 'e', 'n', 'l'};
    int64_t idx_Tmp_1[4] = {'f', 'e', 'm', 'n'};
    TAPP_create_tensor_product(&plan_1, exec, op_A, info_F1, idx_F1_1, op_B, info_T1, idx_T1_1, op_C, info_Tmp, idx_Tmp_1, op_D, info_Tmp, idx_Tmp_1, prec);
    TAPP_error error = TAPP_execute_product(plan_1, exec, 0, (void *)&ALPHA_ONE, (void *)F1_d, (void *)T1_d, (void *)&BETA_ZERO, (void *)Tmp_d, (void *)Tmp_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    int message_len = TAPP_explain_error(error, 0, NULL);
    char *message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    TAPP_tensor_permute plan_2;
    int64_t idx_Z2[4] = {'e', 'f', 'm', 'n'};
    idx_Tmp_1[0] = 'e';
    idx_Tmp_1[1] = 'f';
    idx_Tmp_1[2] = 'm';
    idx_Tmp_1[3] = 'n';
    int64_t idx_Tmp_2[4] = {'e', 'f', 'n', 'm'};
    TAPP_create_tensor_permute(&plan_2, exec, op_A, info_Tmp, idx_Tmp_2, op_B, info_Tmp, idx_Tmp_1, op_C, info_Z2, idx_Z2, prec);
    error = TAPP_execute_permute(plan_2, exec, 0, (void *)&ALPHA_MINUS_ONE, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Tmp_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    cudaFree(F1_d);

    //////////////////////////////////
    // Contractions Z2(e,f,m,n) += 1/2 V2(e,f,d,g) T1(d,g,m,n) 
    //////////////////////////////////
    cudaMalloc(&V2_d, sizeof(double)*num_V2);
    cudaMemcpy(V2_d, V2, num_V2 * sizeof(double), cudaMemcpyHostToDevice);

    TAPP_tensor_product plan_3;
    int64_t idx_V2_3[4] = {'e', 'f', 'd', 'g'};
    int64_t idx_T1_3[4] = {'d', 'g', 'm', 'n'};
    TAPP_create_tensor_product(&plan_3, exec, op_A, info_V2, idx_V2_3, op_B, info_T1, idx_T1_3, op_C, info_Z2, idx_Z2, op_D, info_Z2, idx_Z2, prec);
    error = TAPP_execute_product(plan_3, exec, 0, (void *)&ALPHA_ONE_HALF, (void *)V2_d, (void *)T1_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    cudaFree(V2_d);
    
    //////////////////////////////////
    // Contractions with G1 intermediate: G1(e,f,m,n) = V3(e,l,m,d) T1(f,d,n,l)
    //////////////////////////////////
    cudaMalloc(&V3_d, sizeof(double)*num_V3);
    cudaMemcpy(V3_d, V3, num_V3 * sizeof(double), cudaMemcpyHostToDevice);

    TAPP_tensor_product plan_4;
    int64_t idx_V3_4[4] = {'e', 'l', 'm', 'd'};
    int64_t idx_T1_4[4] = {'f', 'd', 'n', 'l'};
    int64_t idx_G1_4[4] = {'e', 'f', 'm', 'n'};
    TAPP_create_tensor_product(&plan_4, exec, op_A, info_V3, idx_V3_4, op_B, info_T1, idx_T1_4, op_C, info_Tmp, idx_G1_4, op_D, info_Tmp, idx_G1_4, prec);
    error = TAPP_execute_product(plan_4, exec, 0, (void *)&ALPHA_ONE, (void *)V3_d, (void *)T1_d, (void *)&BETA_ZERO, (void *)Tmp_d, (void *)Tmp_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    TAPP_tensor_permute plan_5;
    TAPP_create_tensor_permute(&plan_5, exec, op_A, info_Z2, idx_Z2, op_B, info_Z2, idx_Z2, op_C, info_Z2, idx_Z2, prec);
    error = TAPP_execute_permute(plan_5, exec, 0, (void *)&ALPHA_ONE, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    TAPP_tensor_permute plan_6;
    int64_t idx_G1_6[4] = {'f', 'e', 'm', 'n'};
    TAPP_create_tensor_permute(&plan_6, exec, op_A, info_Z2, idx_G1_6, op_B, info_Z2, idx_Z2, op_C, info_Z2, idx_Z2, prec);
    error = TAPP_execute_permute(plan_6, exec, 0, (void *)&ALPHA_MINUS_ONE, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    error = TAPP_execute_permute(plan_2, exec, 0, (void *)&ALPHA_MINUS_ONE, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    TAPP_tensor_permute plan_7;
    int64_t idx_G1_7[4] = {'f', 'e', 'n', 'm'};
    TAPP_create_tensor_permute(&plan_7, exec, op_A, info_Z2, idx_G1_7, op_B, info_Z2, idx_Z2, op_C, info_Z2, idx_Z2, prec);
    error = TAPP_execute_permute(plan_7, exec, 0, (void *)&ALPHA_ONE, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    cudaFree(V3_d);

    //////////////////////////////////
    // Contractions with I1 intermediate: I1(e,f,n,m) = F2(e,d) T1(f,d,n,m)
    //////////////////////////////////
    cudaMalloc(&F2_d, sizeof(double)*num_F2);
    cudaMemcpy(F2_d, F2, num_F2 * sizeof(double), cudaMemcpyHostToDevice);

    TAPP_tensor_product plan_8;
    int64_t idx_F2_8[2] = {'e', 'd'};
    int64_t idx_T1_8[4] = {'f', 'd', 'n', 'm'};
    TAPP_create_tensor_product(&plan_8, exec, op_A, info_F2, idx_F2_8, op_B, info_T1, idx_T1_8, op_C, info_Tmp, idx_Z2, op_D, info_Tmp, idx_Z2, prec);
    error = TAPP_execute_product(plan_8, exec, 0, (void *)&ALPHA_ONE, (void *)F2_d, (void *)T1_d, (void *)&BETA_ZERO, (void *)Tmp_d, (void *)Tmp_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    error = TAPP_execute_permute(plan_2, exec, 0, (void *)&ALPHA_ONE, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Tmp_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    error = TAPP_execute_permute(plan_7, exec, 0, (void *)&ALPHA_ONE, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Tmp_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    cudaFree(F2_d);

    //////////////////////////////////
    // Contractions with M1 intermediate: M1(e,f,m,n) = T1(e,d,m,n) J1(f,d)
    // with J1(f,d) = V1(l,m,d,e) T1(e,f,l,m)
    //////////////////////////////////
    // J1 vv
    int nmode_J1 = 2;
    int64_t extents_J1[2] = {nvir, nvir};
    int64_t strides_J1[2];
    strides2(extents_J1, strides_J1);
    TAPP_tensor_info info_J1;
    TAPP_create_tensor_info(&info_J1, TAPP_F64, nmode_J1, extents_J1, strides_J1);
    size_t num_J1 = nvir*nvir;
    double *J1_d = nullptr;
    cudaMalloc(&J1_d, sizeof(double)*num_J1);

    TAPP_tensor_product plan_9;
    int64_t idx_V1_9[4] = {'l', 'm', 'd', 'e'};
    int64_t idx_T1_9[4] = {'e', 'f', 'l', 'm'};
    int64_t idx_J1[2] = {'f', 'd'};
    TAPP_create_tensor_product(&plan_9, exec, op_A, info_V1, idx_V1_9, op_B, info_T1, idx_T1_9, op_C, info_J1, idx_J1, op_D, info_J1, idx_J1, prec);
    error = TAPP_execute_product(plan_9, exec, 0, (void *)&ALPHA_ONE, (void *)V1_d, (void *)T1_d, (void *)&BETA_ZERO, (void *)J1_d, (void *)J1_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    TAPP_tensor_product plan_10;
    int64_t idx_T1_10[4] = {'e', 'd', 'm', 'n'};
    TAPP_create_tensor_product(&plan_10, exec, op_A, info_T1, idx_T1_10, op_B, info_J1, idx_J1, op_C, info_Z2, idx_Z2, op_D, info_Z2, idx_Z2, prec);
    error = TAPP_execute_product(plan_10, exec, 0, (void *)&ALPHA_ONE, (void *)T1_d, (void *)J1_d, (void *)&BETA_ZERO, (void *)Tmp_d, (void *)Tmp_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    // Destroy J1 as it is not needed anymore
    TAPP_wait_all(exec);
    TAPP_destroy_tensor_info(info_J1);
    cudaFree(J1_d);

    error = TAPP_execute_permute(plan_5, exec, 0, (void *)&ALPHA_ONE_HALF, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    error = TAPP_execute_permute(plan_6, exec, 0, (void *)&ALPHA_MINUS_ONE_HALF, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    //////////////////////////////////
    // Contractions with Y1 intermediate: Y1(e,f,m,n) = T1(e,d,m,l) X1(l,f,d,n)
    // with X1(l,f,d,n) = V1(l,m,d,e) T1(e,f,m,n)
    //////////////////////////////////
    // X1 ovvo
    int nmode_X1 = 4;
    int64_t extents_X1[4] = {nocc, nvir, nvir, nocc};
    int64_t strides_X1[4];
    strides4(extents_X1, strides_X1);
    TAPP_tensor_info info_X1;
    TAPP_create_tensor_info(&info_X1, TAPP_F64, nmode_X1, extents_X1, strides_X1);
    size_t num_X1 = nvir*nvir*nocc*nocc;
    double *X1_d = nullptr;
    cudaMalloc(&X1_d, sizeof(double)*num_X1);
    
    TAPP_tensor_product plan_11;
    int64_t idx_V1_11[4] = {'l', 'm', 'd', 'e'};
    int64_t idx_T1_11[4] = {'e', 'f', 'm', 'n'};
    int64_t idx_X1[4] = {'l', 'f', 'd', 'n'};
    TAPP_create_tensor_product(&plan_11, exec, op_A, info_V1, idx_V1_11, op_B, info_T1, idx_T1_11, op_C, info_X1, idx_X1, op_D, info_X1, idx_X1, prec);
    error = TAPP_execute_product(plan_11, exec, 0, (void *)&ALPHA_ONE, (void *)V1_d, (void *)T1_d, (void *)&BETA_ZERO, (void *)X1_d, (void *)X1_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    TAPP_tensor_product plan_12;
    int64_t idx_T1_12[4] = {'e', 'd', 'm', 'l'};
    TAPP_create_tensor_product(&plan_12, exec, op_A, info_T1, idx_T1_12, op_B, info_X1, idx_X1, op_C, info_Z2, idx_Z2, op_D, info_Z2, idx_Z2, prec);
    error = TAPP_execute_product(plan_12, exec, 0, (void *)&ALPHA_ONE, (void *)T1_d, (void *)X1_d, (void *)&BETA_ZERO, (void *)Tmp_d, (void *)Tmp_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    // Destroy X1 as it is not needed anymore
    TAPP_wait_all(exec);
    TAPP_destroy_tensor_info(info_X1);
    cudaFree(X1_d);

    error = TAPP_execute_permute(plan_5, exec, 0, (void *)&ALPHA_ONE_HALF, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    error = TAPP_execute_permute(plan_6, exec, 0, (void *)&ALPHA_MINUS_ONE_HALF, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    error = TAPP_execute_permute(plan_2, exec, 0, (void *)&ALPHA_MINUS_ONE_HALF, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    error = TAPP_execute_permute(plan_7, exec, 0, (void *)&ALPHA_ONE_HALF, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    //////////////////////////////////
    // Contractions with G2 intermediate: G2(e,f,m,n) = T1(e,f,m,l) A2(l,n)
    // with A2(l,n) = V1(l,m,d,e) T1(d,e,m,n)
    //////////////////////////////////
    // A2 oo
    int nmode_A2 = 2;
    int64_t extents_A2[2] = {nocc, nocc};
    int64_t strides_A2[2];
    strides2(extents_A2, strides_A2);
    TAPP_tensor_info info_A2;
    TAPP_create_tensor_info(&info_A2, TAPP_F64, nmode_A2, extents_A2, strides_A2);
    size_t num_A2 = nocc*nocc;
    double *A2_d = nullptr;
    cudaMalloc(&A2_d, sizeof(double)*num_A2);
    
    TAPP_tensor_product plan_13;
    int64_t idx_V1_13[4] = {'l', 'm', 'd', 'e'};
    int64_t idx_T1_13[4] = {'d', 'e', 'm', 'n'};
    int64_t idx_A2[2] = {'l', 'n'};
    TAPP_create_tensor_product(&plan_13, exec, op_A, info_V1, idx_V1_13, op_B, info_T1, idx_T1_13, op_C, info_A2, idx_A2, op_D, info_A2, idx_A2, prec);
    error = TAPP_execute_product(plan_13, exec, 0, (void *)&ALPHA_ONE, (void *)V1_d, (void *)T1_d, (void *)&BETA_ZERO, (void *)A2_d, (void *)A2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    TAPP_tensor_product plan_14;
    int64_t idx_T1_14[4] = {'e', 'f', 'm', 'l'};
    TAPP_create_tensor_product(&plan_14, exec, op_A, info_T1, idx_T1_14, op_B, info_A2, idx_A2, op_C, info_Z2, idx_Z2, op_D, info_Z2, idx_Z2, prec);
    error = TAPP_execute_product(plan_14, exec, 0, (void *)&ALPHA_ONE, (void *)T1_d, (void *)A2_d, (void *)&BETA_ZERO, (void *)Tmp_d, (void *)Tmp_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    // Destroy A2 as it is not needed anymore
    TAPP_wait_all(exec);
    TAPP_destroy_tensor_info(info_A2);
    cudaFree(A2_d);

    error = TAPP_execute_permute(plan_5, exec, 0, (void *)&ALPHA_ONE_HALF, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    error = TAPP_execute_permute(plan_2, exec, 0, (void *)&ALPHA_MINUS_ONE_HALF, (void *)Tmp_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    //////////////////////////////////
    // Contractions with E2(l,o,m,n) = V1(l,o,d,e) T1(d,e,m,n)
    // += 2 V4(l,o,m,n)
    //////////////////////////////////
    cudaMalloc(&V4_d, sizeof(double)*num_V4);
    cudaMemcpy(V4_d, V4, num_V4 * sizeof(double), cudaMemcpyHostToDevice);

    // E2 oooo
    int nmode_E2 = 4;
    int64_t extents_E2[4] = {nocc, nocc, nocc, nocc};
    int64_t strides_E2[4];
    strides4(extents_E2, strides_E2);
    TAPP_tensor_info info_E2;
    TAPP_create_tensor_info(&info_E2, TAPP_F64, nmode_E2, extents_E2, strides_E2);
    size_t num_E2 = nocc*nocc*nocc*nocc;
    double *E2_d = nullptr;
    cudaMalloc(&E2_d, sizeof(double)*num_E2);

    TAPP_tensor_product plan_15;
    int64_t idx_V1_15[4] = {'l', 'o', 'd', 'e'};
    int64_t idx_T1_15[4] = {'d', 'e', 'm', 'n'};
    int64_t idx_E2[4] = {'l', 'o', 'm', 'n'};
    TAPP_create_tensor_product(&plan_15, exec, op_A, info_V1, idx_V1_15, op_B, info_T1, idx_T1_15, op_C, info_E2, idx_E2, op_D, info_E2, idx_E2, prec);
    error = TAPP_execute_product(plan_15, exec, 0, (void *)&ALPHA_ONE, (void *)V1_d, (void *)T1_d, (void *)&BETA_ZERO, (void *)E2_d, (void *)E2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    TAPP_tensor_permute plan_16;
    int64_t idx_V4[4] = {'l', 'o', 'm', 'n'};
    TAPP_create_tensor_permute(&plan_16, exec, op_A, info_V4, idx_V4, op_B, info_E2, idx_E2, op_C, info_E2, idx_E2, prec);
    error = TAPP_execute_permute(plan_16, exec, 0, (void *)&ALPHA_TWO, (void *)V4_d, (void *)&BETA_ONE, (void *)E2_d, (void *)E2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    TAPP_tensor_product plan_17;
    int64_t idx_T1_17[4] = {'e', 'f', 'l', 'o'};
    TAPP_create_tensor_product(&plan_17, exec, op_A, info_T1, idx_T1_17, op_B, info_E2, idx_E2, op_C, info_Z2, idx_Z2, op_D, info_Z2, idx_Z2, prec);
    error = TAPP_execute_product(plan_17, exec, 0, (void *)&ALPHA_ONE_QUARTER, (void *)T1_d, (void *)E2_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);
    
    // Destroy E2 as it is not needed anymore
    TAPP_wait_all(exec);
    TAPP_destroy_tensor_info(info_E2);
    cudaFree(E2_d);

    cudaFree(V4_d);

    //////////////////////////////////
    // Last Accumulation with V5
    //////////////////////////////////
    cudaMalloc(&V5_d, sizeof(double)*num_V5);
    cudaMemcpy(V5_d, V5, num_V5 * sizeof(double), cudaMemcpyHostToDevice);

    error = TAPP_execute_permute(plan_5, exec, 0, (void *)&ALPHA_ONE, (void *)V5_d, (void *)&BETA_ONE, (void *)Z2_d, (void *)Z2_d);
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    message_len = TAPP_explain_error(error, 0, NULL);
    message_buff = (char*) malloc((message_len + 1) * sizeof(char));
    TAPP_explain_error(error, message_len + 1, message_buff);
    if(!TAPP_check_success(error)) printf(message_buff);
    free(message_buff);

    TAPP_wait_all(exec);
    name = "Z2";
    printf("Z2 done!\n");

    // Cleanup device data
    cudaFree(T1_d);
    cudaFree(Tmp_d);
    cudaFree(V1_d);
    cudaFree(V5_d);
    cudaFree(Z2_d);
    
    // Cleanup plans
    TAPP_wait_all(exec);
    TAPP_destroy_tensor_product(plan_1);  // femn = lm fenl
    TAPP_destroy_tensor_permute(plan_2);  // efmn = efnm + efmn
    TAPP_destroy_tensor_product(plan_3);  // efmn = efdg dgmn 
    TAPP_destroy_tensor_product(plan_4);  // efmn = elmd fdnl
    TAPP_destroy_tensor_permute(plan_5);  // efmn = efmn + efmn
    TAPP_destroy_tensor_permute(plan_6);  // efmn = femn + efmn
    TAPP_destroy_tensor_permute(plan_7);  // efmn = fenm + efmn
    TAPP_destroy_tensor_product(plan_8);  // efmn = ed fdnm
    TAPP_destroy_tensor_product(plan_9);  // fd = lmde eflm
    TAPP_destroy_tensor_product(plan_10); // efmn = dmn fd
    TAPP_destroy_tensor_product(plan_11); // lfdn = lmde efmn
    TAPP_destroy_tensor_product(plan_12); // efmn = edml lfdn
    TAPP_destroy_tensor_product(plan_13); // ln = lmde demn
    TAPP_destroy_tensor_product(plan_14); // efmn = efml ln
    TAPP_destroy_tensor_product(plan_15); // efmn = efml ln
    TAPP_destroy_tensor_permute(plan_16); // lomn = lomn + lomn
    TAPP_destroy_tensor_product(plan_17); 

    // Cleanup tensors
    printf("\nCleaning up...\n");
    // V1 oovv
    TAPP_destroy_tensor_info(info_V1);
    // T1 vvoo
    TAPP_destroy_tensor_info(info_T1);
    // F1 oo
    TAPP_destroy_tensor_info(info_F1);
    // V2 vvvv
    TAPP_destroy_tensor_info(info_V2);
    // V3 voov
    TAPP_destroy_tensor_info(info_V3);
    // F2 vv
    TAPP_destroy_tensor_info(info_F2);
    // V4 oooo
    TAPP_destroy_tensor_info(info_V4);
    // V5 vvoo
    TAPP_destroy_tensor_info(info_V5);
    // Z2 vvoo
    TAPP_destroy_tensor_info(info_Z2);
    // Tmp vvoo
    TAPP_destroy_tensor_info(info_Tmp);
    
    TAPP_destroy_executor(exec);
    TAPP_shutdown();
    return 0;
}

void strides4(const int64_t e[4], int64_t s[4]){
	 s[0] = 1; s[1] = e[0]; s[2] = e[0] * e[1]; s[3] = e[0] * e[1] * e[2];
}
void strides3(const int64_t e[3], int64_t s[3]){
	 s[0] = 1; s[1] = e[0]; s[2] = e[0] * e[1];
}
void strides2(const int64_t e[2], int64_t s[2]){
	 s[0] = 1; s[1] = e[0];
}
void set_tensor_to_zero(double* T, size_t n){
	#pragma omp for schedule(static)
	for(size_t i=0; i<n; i++){
		T[i] = 0.0;
	}

}

void save_tensor(double* T, size_t n, std::string name){
    FILE *file = fopen((name+".bin").c_str(), "wb");
    if (!file){
      printf("Error in save_tensor\n");
      return; // Handle error appropriately
    }
    fwrite(reinterpret_cast<const char*>(&n), sizeof(size_t), 1, file);
    fwrite(T, sizeof(double), n, file);
    fclose(file);
}
void read_tensor(double* T, size_t n, std::string name, int64_t nocc, int64_t nvir){

	size_t nread;
	std::string FILE_NAME = "../build_tblis/tblis-reference-"+std::to_string(nocc)+"-"+std::to_string(nvir)+"/"+name+".bin";
	FILE *file = fopen((FILE_NAME).c_str(), "rb");
	if (!file) {
            std::cerr << "Error: Could not open file " << FILE_NAME << "\n";
            return;
        }
	fread(&nread, sizeof(size_t), 1, file);
	fread(T, sizeof(double), n, file);
	fclose(file);
        if(n!=nread) printf("Error when reading file...\n");	
}

