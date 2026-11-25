/*
 * Demo for the cuBLAS/TTGT TAPP bindings.
 *
 * Mirrors test/cutensor_demo.cpp but is scoped to the operations the vendored
 * TTGT (transpose-transpose-GEMM-transpose) back-end supports: plain
 * contractions over dense, generalized-column-major tensors with identity
 * element-wise ops. Each case runs on device memory (the default) and checks
 * the result against a host reference.
 */

#include <tapp.h>
#include <tapp_cublas.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <complex>

void contraction();
void contraction_beta();
void complex_contraction();
void outer_product();
void full_contraction();
void attributes();
void permute();

void check_status(TAPP_status status)
{
    TAPP_error op_error = TAPP_status_get_error(status);
    if (!TAPP_check_success(op_error))
    {
        int len = TAPP_explain_error(op_error, 0, NULL);
        char *buff = (char*)malloc((len + 1) * sizeof(char));
        TAPP_explain_error(op_error, len + 1, buff);
        printf("Operation failed: %s\n", buff);
        free(buff);
    }
}

void report_error(TAPP_error error)
{
    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n");
    if (!TAPP_check_success(error))
    {
        int len = TAPP_explain_error(error, 0, NULL);
        char *buff = (char*)malloc((len + 1) * sizeof(char));
        TAPP_explain_error(error, len + 1, buff);
        printf("%s\n", buff);
        free(buff);
    }
}

int main(int argc, char const *argv[])
{
    (void)argc;
    (void)argv;
    printf("Contraction (ik,kj->ij, beta=0): \n");
    contraction();
    printf("Contraction with beta != 0: \n");
    contraction_beta();
    printf("Complex contraction (C32): \n");
    complex_contraction();
    printf("Attributes: \n");
    attributes();
    printf("Outer product: \n");
    outer_product();
    printf("Full contraction: \n");
    full_contraction();
    printf("Permute: \n");
    permute();
    return 0;
}

// D_ij = alpha * sum_k A_ik B_kj + beta * C_ij, all column-major dense.
void contraction()
{
    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    const int ni = 2, nk = 3, nj = 2;

    int nmode_A = 2;
    int64_t extents_A[2] = {ni, nk};
    int64_t strides_A[2] = {1, ni};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {nk, nj};
    int64_t strides_B[2] = {1, nk};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_D = 2;
    int64_t extents_D[2] = {ni, nj};
    int64_t strides_D[2] = {1, ni};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F32, nmode_D, extents_D, strides_D);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F32, nmode_D, extents_D, strides_D);

    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_D[2] = {'i', 'j'};

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, exec,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_D,
        TAPP_IDENTITY, info_D, idx_D,
        TAPP_DEFAULT_PREC);

    TAPP_status status;

    float alpha = 1.0f;
    float beta = 0.0f;
    float A[6] = {1, 2,  3, 4,  5, 6};     // A(i,k) at i + k*ni
    float B[6] = {1, 0, 1,  0, 1, 0};      // B(k,j) at k + j*nk
    float C[4] = {0, 0, 0, 0};
    float D[4] = {0, 0, 0, 0};

    void *A_d, *B_d, *C_d, *D_d;
    cudaMalloc(&A_d, sizeof(A));
    cudaMalloc(&B_d, sizeof(B));
    cudaMalloc(&C_d, sizeof(C));
    cudaMalloc(&D_d, sizeof(D));
    cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice);

    TAPP_error error = TAPP_execute_product(plan, exec, &status,
        &alpha, A_d, B_d, &beta, C_d, D_d);
    report_error(error);

    TAPP_wait_all(exec);
    check_status(status);
    cudaMemcpy(D, D_d, sizeof(D), cudaMemcpyDeviceToHost);

    // Host reference.
    float ref[4];
    for (int j = 0; j < nj; j++)
        for (int i = 0; i < ni; i++)
        {
            float acc = 0.0f;
            for (int k = 0; k < nk; k++)
                acc += A[i + k * ni] * B[k + j * nk];
            ref[i + j * ni] = alpha * acc + beta * C[i + j * ni];
        }

    bool ok = true;
    for (int idx = 0; idx < 4; idx++)
    {
        printf("\tD[%d] = %.3f (ref %.3f)\n", idx, D[idx], ref[idx]);
        if (std::fabs(D[idx] - ref[idx]) > 1e-4f) ok = false;
    }
    printf(ok ? "\tPASS\n" : "\tFAIL\n");

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_status(status);
    TAPP_destroy_executor(exec);
}

// Same contraction but with beta != 0, exercising the C-into-D staging path.
void contraction_beta()
{
    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    const int ni = 2, nk = 2, nj = 2;

    int64_t extents_A[2] = {ni, nk};
    int64_t strides_A[2] = {1, ni};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_F64, 2, extents_A, strides_A);

    int64_t extents_B[2] = {nk, nj};
    int64_t strides_B[2] = {1, nk};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, 2, extents_B, strides_B);

    int64_t extents_D[2] = {ni, nj};
    int64_t strides_D[2] = {1, ni};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, 2, extents_D, strides_D);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, 2, extents_D, strides_D);

    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_D[2] = {'i', 'j'};

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, exec,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_D,
        TAPP_IDENTITY, info_D, idx_D,
        TAPP_DEFAULT_PREC);

    TAPP_status status;

    double alpha = 2.0;
    double beta = 0.5;
    double A[4] = {1, 2, 3, 4};
    double B[4] = {1, 0, 0, 1}; // identity
    double C[4] = {10, 20, 30, 40};
    double D[4] = {0, 0, 0, 0};

    void *A_d, *B_d, *C_d, *D_d;
    cudaMalloc(&A_d, sizeof(A));
    cudaMalloc(&B_d, sizeof(B));
    cudaMalloc(&C_d, sizeof(C));
    cudaMalloc(&D_d, sizeof(D));
    cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice);

    TAPP_error error = TAPP_execute_product(plan, exec, &status,
        &alpha, A_d, B_d, &beta, C_d, D_d);
    report_error(error);

    TAPP_wait_all(exec);
    check_status(status);
    cudaMemcpy(D, D_d, sizeof(D), cudaMemcpyDeviceToHost);

    double ref[4];
    for (int j = 0; j < nj; j++)
        for (int i = 0; i < ni; i++)
        {
            double acc = 0.0;
            for (int k = 0; k < nk; k++)
                acc += A[i + k * ni] * B[k + j * nk];
            ref[i + j * ni] = alpha * acc + beta * C[i + j * ni];
        }

    bool ok = true;
    for (int idx = 0; idx < 4; idx++)
    {
        printf("\tD[%d] = %.3f (ref %.3f)\n", idx, D[idx], ref[idx]);
        if (std::fabs(D[idx] - ref[idx]) > 1e-9) ok = false;
    }
    printf(ok ? "\tPASS\n" : "\tFAIL\n");

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_status(status);
    TAPP_destroy_executor(exec);
}

// Single-precision complex contraction.
void complex_contraction()
{
    using cf = std::complex<float>;
    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    const int ni = 2, nk = 2, nj = 2;

    int64_t extents_A[2] = {ni, nk};
    int64_t strides_A[2] = {1, ni};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, 2, extents_A, strides_A);

    int64_t extents_B[2] = {nk, nj};
    int64_t strides_B[2] = {1, nk};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, 2, extents_B, strides_B);

    int64_t extents_D[2] = {ni, nj};
    int64_t strides_D[2] = {1, ni};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, 2, extents_D, strides_D);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C32, 2, extents_D, strides_D);

    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_D[2] = {'i', 'j'};

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, exec,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_D,
        TAPP_IDENTITY, info_D, idx_D,
        TAPP_DEFAULT_PREC);

    TAPP_status status;

    cf alpha(1.0f, 0.0f);
    cf beta(0.0f, 0.0f);
    cf A[4] = {cf(1, 1), cf(2, 0), cf(0, 1), cf(1, -1)};
    cf B[4] = {cf(1, 0), cf(0, 1), cf(1, 1), cf(2, 0)};
    cf C[4] = {cf(0, 0), cf(0, 0), cf(0, 0), cf(0, 0)};
    cf D[4] = {cf(0, 0), cf(0, 0), cf(0, 0), cf(0, 0)};

    void *A_d, *B_d, *C_d, *D_d;
    cudaMalloc(&A_d, sizeof(A));
    cudaMalloc(&B_d, sizeof(B));
    cudaMalloc(&C_d, sizeof(C));
    cudaMalloc(&D_d, sizeof(D));
    cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice);

    TAPP_error error = TAPP_execute_product(plan, exec, &status,
        &alpha, A_d, B_d, &beta, C_d, D_d);
    report_error(error);

    TAPP_wait_all(exec);
    check_status(status);
    cudaMemcpy(D, D_d, sizeof(D), cudaMemcpyDeviceToHost);

    cf ref[4];
    for (int j = 0; j < nj; j++)
        for (int i = 0; i < ni; i++)
        {
            cf acc(0, 0);
            for (int k = 0; k < nk; k++)
                acc += A[i + k * ni] * B[k + j * nk];
            ref[i + j * ni] = alpha * acc + beta * C[i + j * ni];
        }

    bool ok = true;
    for (int idx = 0; idx < 4; idx++)
    {
        printf("\tD[%d] = (%.3f, %.3f) (ref (%.3f, %.3f))\n",
            idx, D[idx].real(), D[idx].imag(), ref[idx].real(), ref[idx].imag());
        if (std::abs(D[idx] - ref[idx]) > 1e-4f) ok = false;
    }
    printf(ok ? "\tPASS\n" : "\tFAIL\n");

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_status(status);
    TAPP_destroy_executor(exec);
}

// Sets, reads back and validates the cuBLAS attribute keys, then creates an executor from them
void attributes()
{
    TAPP_attr attr;
    TAPP_create_attr(&attr);

    size_t size;
    void* value;

    TAPP_emulation_strategy strategy = TAPP_EMULATION_STRATEGY_PERFORMANT;
    TAPP_error error = TAPP_attr_set(attr, TAPP_CUBLAS_EMULATION_STRATEGY, sizeof(strategy), &strategy);
    printf("\tset TAPP_CUBLAS_EMULATION_STRATEGY (should be 1): %i\n", TAPP_check_success(error));

    TAPP_emulation_mantissa_control mantissa_control = TAPP_EMULATION_MANTISSA_CONTROL_FIXED;
    error = TAPP_attr_set(attr, TAPP_CUBLAS_EMULATION_MANTISSA_CONTROL, sizeof(mantissa_control), &mantissa_control);
    printf("\tset TAPP_CUBLAS_EMULATION_MANTISSA_CONTROL (should be 1): %i\n", TAPP_check_success(error));

    TAPP_executor exec;
    error = TAPP_create_executor(&exec, attr);
    printf("\tcreate executor from attributes (should be 1): %i\n", TAPP_check_success(error));
    if (TAPP_check_success(error)) TAPP_destroy_executor(exec);

    TAPP_destroy_attr(attr);
}
// Outer product test
void outer_product(){
    using cf = std::complex<float>;
    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    const int ni = 2, nj = 3, nk = 4, nl =5;

    int64_t extents_A[3] = {ni, nj, nk};
    int64_t strides_A[3] = {1, ni, ni*nj};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, 3, extents_A, strides_A);

    int64_t extents_B[1] = {nl};
    int64_t strides_B[1] = {1};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, 1, extents_B, strides_B);

    int64_t sizeD = ni* nj* nl* nk;
    int64_t extents_D[4] = {ni, nj, nl, nk};
    int64_t strides_D[4] = {1, ni, ni*nj, ni*nj*nl};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, 4, extents_D, strides_D);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C32, 4, extents_D, strides_D);

    int64_t idx_A[3] = {'i', 'j', 'k'};
    int64_t idx_B[1] = {'l'};
    int64_t idx_D[4] = {'i', 'j', 'l', 'k'};

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, exec,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_D,
        TAPP_IDENTITY, info_D, idx_D,
        TAPP_DEFAULT_PREC);

    TAPP_status status;

    cf alpha(1.0f, 0.0f);
    cf beta(0.0f, 0.0f);
    cf A[24] = {
         cf(0.24,0.312),
         cf(0.889,-0.091),
         cf(0.458,0.346),
         cf(-0.22,-0.884),
         cf(0.128,0.781),
         cf(-0.665,-0.883),
         cf(0.251,0.552),
         cf(0.276,0.86),
         cf(-0.609,0.707),
         cf(0.137,0.136),
         cf(0.787,-0.542),
         cf(-0.259,-0.361),
         cf(-0.714,-0.684),
         cf(-0.462,-0.4),
         cf(-0.872,-0.528),
         cf(-0.88,-0.616),
         cf(-0.678,0.221),
         cf(-0.232,-0.347),
         cf(0.421,-0.799),
         cf(-0.312,-0.311),
         cf(0.591,-0.185),
         cf(-0.771,0.518),
         cf(0.878,0.966),
         cf(-0.602,0.681)
    };
    cf B[5] = {
         cf(0.014,-0.502),
         cf(-0.399,-0.726),
         cf(-0.33,-0.296),
         cf(-0.869,0.259),
         cf(0.064,0.731)
    };

    cf C[sizeD];
    cf D[sizeD];
    for(int i=0;i<sizeD;i++){
        C[i] = cf(0, 0);
        D[i] = cf(0, 0);
    }

    void *A_d, *B_d, *C_d, *D_d;
    cudaMalloc(&A_d, sizeof(A));
    cudaMalloc(&B_d, sizeof(B));
    cudaMalloc(&C_d, sizeof(C));
    cudaMalloc(&D_d, sizeof(D));
    cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice);

    TAPP_error error = TAPP_execute_product(plan, exec, &status,
        &alpha, A_d, B_d, &beta, C_d, D_d);
    report_error(error);

    TAPP_wait_all(exec);
    check_status(status);
    cudaMemcpy(D, D_d, sizeof(D), cudaMemcpyDeviceToHost);

    cf ref[sizeD];
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < nj; j++)
            for (int l = 0; l < nl; l++)
                for (int k = 0; k < nk; k++){
                    int indA = i + j*ni + k*ni*nj;
                    int indB = l;
                    int ind = i + j*ni + l*ni*nj + k*ni*nj*nl;
                    ref[ind] = A[indA] * B[indB];
                }

    bool ok = true;
    for (int idx = 0; idx < sizeD; idx++)
    {
        printf("\tD[%d] = (%.3f, %.3f) (ref (%.3f, %.3f))\n",
            idx, D[idx].real(), D[idx].imag(), ref[idx].real(), ref[idx].imag());
        if (std::abs(D[idx] - ref[idx]) > 1e-4f) ok = false;
    }
    printf(ok ? "\tPASS\n" : "\tFAIL\n");

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_status(status);
    TAPP_destroy_executor(exec);
}

void full_contraction(){
    using cf = std::complex<float>;
    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    const int ni = 2, nj = 3, na = 4, nb =5;

    int64_t extents_A[4] = {ni, nj, na, nb};
    int64_t strides_A[4] = {1, ni, ni*nj, ni*nj*na};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, 4, extents_A, strides_A);

    int64_t extents_B[4] = {na, nb, ni, nj};
    int64_t strides_B[4] = {1, na, na*nb, na*nb*ni};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, 4, extents_B, strides_B);

    int64_t extents_D[0];
    int64_t strides_D[1] = {1};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, 0, extents_D, strides_D);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_C32, 0, extents_D, strides_D);

    int64_t idx_A[4] = {'i', 'j', 'a', 'b'};
    int64_t idx_B[4] = {'a', 'b', 'i', 'j'};
    int64_t idx_D[0];

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, exec,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_D,
        TAPP_IDENTITY, info_D, idx_D,
        TAPP_DEFAULT_PREC);

    TAPP_status status;

    cf alpha(1.0f, 0.0f);
    cf beta(0.0f, 0.0f);
    cf A[120] = {
        cf(-0.104,-0.652),cf(0.568,0.976),cf(0.434,0.753),
        cf(-0.387,0.867),cf(-0.16,0.84),cf(-0.987,0.06),
        cf(0.185,0.642),cf(0.573,0.365),cf(0.281,0.22),
        cf(0.277,-0.334),cf(0.953,-0.0),cf(0.618,-0.063),
        cf(-0.959,0.274),cf(0.106,-0.478),cf(0.691,0.299),
        cf(-0.365,-0.85),cf(-0.46,-0.829),cf(0.751,0.667),
        cf(0.607,-0.541),cf(-0.46,0.251),cf(0.248,-0.677),
        cf(0.703,0.076),cf(0.071,0.559),cf(-0.291,0.35),
        cf(-0.88,-0.422),cf(0.964,0.144),cf(0.266,-0.996),
        cf(0.858,-0.424),cf(0.593,0.373),cf(-0.71,0.51),
        cf(-0.749,0.25),cf(0.042,-0.122),cf(0.187,-0.905),
        cf(-0.771,-0.822),cf(-0.797,-0.169),cf(0.001,-0.17),
        cf(-0.348,0.895),cf(-0.872,-0.012),cf(0.865,0.727),
        cf(-0.604,-0.337),cf(-0.133,-0.913),cf(0.697,-0.695),
        cf(0.773,0.091),cf(0.249,0.495),cf(0.087,-0.809),
        cf(0.83,-0.791),cf(0.69,0.014),cf(0.028,-0.486),
        cf(0.122,-0.46),cf(0.203,0.265),cf(-0.309,-0.35),
        cf(-0.687,0.843),cf(0.893,-0.437),cf(-0.409,-0.319),
        cf(0.251,-0.191),cf(0.395,0.867),cf(-0.107,-0.064),
        cf(-0.861,-0.553),cf(0.458,-0.441),cf(0.17,0.189),
        cf(-0.879,-0.055),cf(0.095,-0.134),cf(0.217,-0.367),
        cf(-0.113,0.662),cf(-0.486,-0.525),cf(-0.501,0.433),
        cf(-0.423,-0.351),cf(0.005,-0.533),cf(0.107,-0.402),
        cf(-0.986,0.549),cf(0.492,-0.099),cf(-0.825,0.851),
        cf(0.453,-0.457),cf(0.282,0.679),cf(0.63,0.488),
        cf(0.743,-0.504),cf(-0.514,0.835),cf(-0.178,-0.188),
        cf(-0.778,-0.988),cf(-0.584,0.546),cf(0.134,-0.675),
        cf(-0.974,0.051),cf(-0.792,-0.569),cf(0.18,0.983),
        cf(-0.991,-0.525),cf(0.009,-0.894),cf(0.562,-0.251),
        cf(0.05,0.779),cf(0.367,0.706),cf(0.646,-0.054),
        cf(-0.901,0.642),cf(0.869,0.209),cf(0.828,-0.256),
        cf(0.192,-0.091),cf(0.492,-0.021),cf(0.975,0.299),
        cf(-0.495,-0.774),cf(0.956,-0.883),cf(0.617,-0.799),
        cf(0.022,0.143),cf(-0.06,-0.711),cf(0.364,-0.922),
        cf(-0.098,0.838),cf(-0.098,0.856),cf(0.384,0.707),
        cf(-0.109,-0.393),cf(-0.945,-0.776),cf(0.529,0.207),
        cf(0.903,-0.065),cf(0.516,-0.274),cf(0.134,-0.486),
        cf(0.693,0.558),cf(-0.723,0.487),cf(-0.219,0.497),
        cf(-0.918,0.403),cf(-0.305,0.545),cf(0.499,0.52),
        cf(0.736,0.45),cf(0.246,0.665),cf(0.63,-0.938)
    };
    cf B[120] = {
        cf(-0.768,0.08),cf(0.758,-0.488),cf(-0.013,-0.647),
        cf(0.446,0.87),cf(-0.053,-0.471),cf(-0.197,-0.501),
        cf(0.914,0.584),cf(-0.467,-0.448),cf(-0.507,0.346),
        cf(-0.89,0.789),cf(0.955,0.864),cf(-0.991,-0.037),
        cf(0.71,-0.546),cf(-0.056,0.905),cf(-0.378,0.133),
        cf(-0.055,-0.359),cf(-0.202,0.094),cf(0.601,-0.074),
        cf(-0.367,0.688),cf(-0.853,-0.365),cf(0.772,-0.619),
        cf(-0.97,-0.806),cf(-0.677,0.327),cf(0.011,-0.492),
        cf(-0.655,-0.453),cf(0.521,0.954),cf(0.542,-0.819),
        cf(0.041,0.407),cf(0.237,-0.543),cf(-0.727,-0.824),
        cf(0.605,-0.66),cf(0.107,-0.351),cf(-0.225,-0.671),
        cf(0.248,-0.6),cf(0.59,-0.758),cf(0.659,0.66),
        cf(0.252,-0.229),cf(0.749,0.237),cf(-0.63,0.964),
        cf(-0.043,-0.433),cf(-0.504,0.312),cf(-0.798,0.892),
        cf(0.791,-0.566),cf(-0.732,0.115),cf(0.591,0.452),
        cf(0.134,-0.189),cf(-0.041,0.936),cf(-0.017,-0.667),
        cf(-0.78,0.898),cf(0.699,-0.185),cf(-0.266,-0.999),
        cf(0.081,0.778),cf(0.671,-0.882),cf(-0.607,0.036),
        cf(-0.803,0.42),cf(0.549,0.175),cf(-0.48,0.193),
        cf(0.613,-0.016),cf(0.921,0.281),cf(0.113,0.259),
        cf(0.491,0.817),cf(0.854,-0.697),cf(0.276,0.045),
        cf(0.731,-0.457),cf(0.614,0.901),cf(-0.614,-0.71),
        cf(-0.304,-0.349),cf(-0.765,-0.009),cf(-0.921,-0.202),
        cf(-0.154,0.973),cf(-0.934,-0.964),cf(-0.303,-0.528),
        cf(0.847,0.12),cf(-0.702,0.329),cf(-0.759,-0.918),
        cf(0.805,0.406),cf(-0.461,0.973),cf(-0.818,-0.141),
        cf(0.85,0.146),cf(0.319,0.452),cf(0.401,-0.76),
        cf(-0.483,0.85),cf(-0.502,-0.923),cf(0.695,0.378),
        cf(-0.768,0.672),cf(0.13,0.669),cf(-0.895,-0.674),
        cf(0.699,0.076),cf(0.534,0.122),cf(0.711,0.018),
        cf(-0.237,-0.694),cf(0.238,0.563),cf(-0.658,0.133),
        cf(-0.989,0.84),cf(-0.024,0.82),cf(-0.49,-0.415),
        cf(-0.101,0.136),cf(-0.811,-0.603),cf(-0.444,0.21),
        cf(0.1,-0.341),cf(-0.543,0.906),cf(0.572,0.652),
        cf(0.211,0.26),cf(0.926,0.735),cf(-0.049,0.754),
        cf(0.523,0.014),cf(-0.214,0.352),cf(0.035,0.069),
        cf(-0.57,-0.349),cf(-0.795,-0.327),cf(0.16,0.834),
        cf(0.902,-0.917),cf(-0.297,-0.933),cf(0.325,-0.443),
        cf(0.099,0.941),cf(-0.159,-0.801),cf(-0.211,-0.028),
        cf(-0.517,0.559),cf(-0.347,-0.527),cf(0.88,-0.278)
    };

    cf C[1] = {cf(0, 0)};
    cf D[1] = {cf(0, 0)};

    void *A_d, *B_d, *C_d, *D_d;
    cudaMalloc(&A_d, sizeof(A));
    cudaMalloc(&B_d, sizeof(B));
    cudaMalloc(&C_d, sizeof(C));
    cudaMalloc(&D_d, sizeof(D));
    cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(B), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, sizeof(C), cudaMemcpyHostToDevice);

    TAPP_error error = TAPP_execute_product(plan, exec, &status,
        &alpha, A_d, B_d, &beta, C_d, D_d);
    report_error(error);

    TAPP_wait_all(exec);
    check_status(status);
    cudaMemcpy(D, D_d, sizeof(D), cudaMemcpyDeviceToHost);

    cf ref[1] = {cf{10.541134999999999, 9.54222}};

    bool ok = true;
    for (int idx = 0; idx < 1; idx++)
    {
        printf("\tD[%d] = (%.3f, %.3f) (ref (%.3f, %.3f))\n",
            idx, D[idx].real(), D[idx].imag(), ref[idx].real(), ref[idx].imag());
        if (std::abs(D[idx] - ref[idx]) > 1e-4f) ok = false;
    }
    printf(ok ? "\tPASS\n" : "\tFAIL\n");

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d); cudaFree(D_d);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_status(status);
    TAPP_destroy_executor(exec);
}

// Test permute opeartion for cuBLAS backend
void permute()
{
    using cf = std::complex<float>;
    TAPP_executor exec;
    TAPP_create_executor(&exec, 0);

    const int ni = 2, nk = 3, nj = 2;

    int nmode_A = 3;
    int64_t extents_A[3] = {nj, ni, nk};
    int64_t strides_A[3] = {1, nj, nj*ni};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, TAPP_C32, nmode_A, extents_A, strides_A);

    int nmode_B = 3;
    int64_t extents_B[3] = {ni, nk, nj};
    int64_t strides_B[3] = {1, ni, nk*nj};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_C32, nmode_B, extents_B, strides_B);

    int nmode_C = 3;
    int64_t extents_C[3] = {ni, nk, nj};
    int64_t strides_C[3] = {1, ni, nk*nj};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_C32, nmode_C, extents_C, strides_C);

    int64_t idx_A[3] = {'j', 'i', 'k'};
    int64_t idx_B[3] = {'i', 'k', 'j'};
    int64_t idx_C[3] = {'i', 'k', 'j'};

    TAPP_tensor_permute plan;
    TAPP_error error = TAPP_create_tensor_permute(&plan, exec,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_C,
        TAPP_DEFAULT_PREC);
    report_error(error);

    TAPP_status status;

    cf alpha(1.0f, 0.0f);
    cf beta(1.0f, 0.0f);
    cf A[12] = {
        cf(-0.104,-0.652),cf(0.568,0.976),cf(0.434,0.753),
        cf(-0.387,0.867),cf(-0.16,0.84),cf(-0.987,0.06),
        cf(0.185,0.642),cf(0.573,0.365),cf(0.281,0.22),
        cf(0.277,-0.334),cf(0.953,-0.0),cf(0.618,-0.063)
    };
    cf B[12] = {
        cf(-0.768,0.08),cf(0.758,-0.488),cf(-0.013,-0.647),
        cf(0.446,0.87),cf(-0.053,-0.471),cf(-0.197,-0.501),
        cf(0.914,0.584),cf(-0.467,-0.448),cf(-0.507,0.346),
        cf(-0.89,0.789),cf(0.955,0.864),cf(-0.991,-0.037)
    };
    cf C[12];

    void *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, sizeof(A));
    cudaMalloc(&B_d, sizeof(B));
    cudaMalloc(&C_d, sizeof(C));

    cudaMemcpy(A_d, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(B), cudaMemcpyHostToDevice);

    error = TAPP_execute_permute(plan, exec, &status,
        &alpha, A_d, &beta, B_d, C_d);
    report_error(error);

    TAPP_wait_all(exec);
    check_status(status);
    cudaMemcpy(C, C_d, sizeof(C), cudaMemcpyDeviceToHost);

    cf ref[12];
    for(int i=0; i<ni; i++)
    for(int k=0; k<nk; k++)
    for(int j=0; j<nj; j++){
       int ind_A_ = j + i*nj + k*nj*ni;
       int ind = i + k*ni + j*ni*nk;
       ref[ind] = alpha*A[ind_A_]+beta*B[ind];
    }

    bool ok = true;
    for (int idx = 0; idx < 12; idx++)
    {
        printf("\tC[%d] = (%.3f, %.3f) (ref (%.3f, %.3f))\n",
            idx, C[idx].real(), C[idx].imag(), ref[idx].real(), ref[idx].imag());
        if (std::abs(C[idx] - ref[idx]) > 1e-4f) ok = false;
    }
    printf(ok ? "\tPASS\n" : "\tFAIL\n");

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    TAPP_destroy_tensor_permute(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_status(status);
    TAPP_destroy_executor(exec);
}

