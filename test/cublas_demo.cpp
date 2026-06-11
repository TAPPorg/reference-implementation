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

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <complex>

void contraction();
void contraction_beta();
void complex_contraction();

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
    return 0;
}

// D_ij = alpha * sum_k A_ik B_kj + beta * C_ij, all column-major dense.
void contraction()
{
    TAPP_handle handle;
    TAPP_create_handle(&handle);

    const int ni = 2, nk = 3, nj = 2;

    int nmode_A = 2;
    int64_t extents_A[2] = {ni, nk};
    int64_t strides_A[2] = {1, ni};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, handle, TAPP_F32, nmode_A, extents_A, strides_A);

    int nmode_B = 2;
    int64_t extents_B[2] = {nk, nj};
    int64_t strides_B[2] = {1, nk};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, handle, TAPP_F32, nmode_B, extents_B, strides_B);

    int nmode_D = 2;
    int64_t extents_D[2] = {ni, nj};
    int64_t strides_D[2] = {1, ni};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, handle, TAPP_F32, nmode_D, extents_D, strides_D);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, handle, TAPP_F32, nmode_D, extents_D, strides_D);

    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_D[2] = {'i', 'j'};

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, handle,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_D,
        TAPP_IDENTITY, info_D, idx_D,
        TAPP_DEFAULT_PREC);

    TAPP_executor exec;
    TAPP_create_executor(&exec);
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

    TAPP_executor_wait(exec);
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
    TAPP_destroy_handle(handle);
}

// Same contraction but with beta != 0, exercising the C-into-D staging path.
void contraction_beta()
{
    TAPP_handle handle;
    TAPP_create_handle(&handle);

    const int ni = 2, nk = 2, nj = 2;

    int64_t extents_A[2] = {ni, nk};
    int64_t strides_A[2] = {1, ni};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, handle, TAPP_F64, 2, extents_A, strides_A);

    int64_t extents_B[2] = {nk, nj};
    int64_t strides_B[2] = {1, nk};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, handle, TAPP_F64, 2, extents_B, strides_B);

    int64_t extents_D[2] = {ni, nj};
    int64_t strides_D[2] = {1, ni};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, handle, TAPP_F64, 2, extents_D, strides_D);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, handle, TAPP_F64, 2, extents_D, strides_D);

    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_D[2] = {'i', 'j'};

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, handle,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_D,
        TAPP_IDENTITY, info_D, idx_D,
        TAPP_DEFAULT_PREC);

    TAPP_executor exec;
    TAPP_create_executor(&exec);
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

    TAPP_executor_wait(exec);
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
    TAPP_destroy_handle(handle);
}

// Single-precision complex contraction.
void complex_contraction()
{
    using cf = std::complex<float>;
    TAPP_handle handle;
    TAPP_create_handle(&handle);

    const int ni = 2, nk = 2, nj = 2;

    int64_t extents_A[2] = {ni, nk};
    int64_t strides_A[2] = {1, ni};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, handle, TAPP_C32, 2, extents_A, strides_A);

    int64_t extents_B[2] = {nk, nj};
    int64_t strides_B[2] = {1, nk};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, handle, TAPP_C32, 2, extents_B, strides_B);

    int64_t extents_D[2] = {ni, nj};
    int64_t strides_D[2] = {1, ni};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, handle, TAPP_C32, 2, extents_D, strides_D);
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, handle, TAPP_C32, 2, extents_D, strides_D);

    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_D[2] = {'i', 'j'};

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, handle,
        TAPP_IDENTITY, info_A, idx_A,
        TAPP_IDENTITY, info_B, idx_B,
        TAPP_IDENTITY, info_C, idx_D,
        TAPP_IDENTITY, info_D, idx_D,
        TAPP_DEFAULT_PREC);

    TAPP_executor exec;
    TAPP_create_executor(&exec);
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

    TAPP_executor_wait(exec);
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
    TAPP_destroy_handle(handle);
}
