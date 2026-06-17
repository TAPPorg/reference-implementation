// CCSD bottleneck contractions, timed via the TAPP API on the GPU.
// Allocation/fill are outside the timed loop.

#include <tapp.h>
#include <cuda_runtime.h>

#include <complex>
#include <cstdint>
#include <cstdio>
#include <string>

// Storage and compute precision. Change here to experiment (e.g. TAPP_F64 +
// TAPP_F_7_DIGITS for emulated reduced precision).
using Scalar = std::complex<double>;
static const TAPP_datatype STORAGE = TAPP_C64;
static const TAPP_prectype PREC = TAPP_DEFAULT_PREC;

static const int ITERS = 10;

// col-major strides for a 4-mode dense tensor
static void strides4(const int64_t e[4], int64_t s[4])
{
    s[0] = 1; s[1] = e[0]; s[2] = e[0] * e[1]; s[3] = e[0] * e[1] * e[2];
}

// PPL: D_abij = 0.5 * A_abcd * B_cdij + C_abij
static void contraction_PPL(int nocc, int nvirt)
{
    int64_t eA[4] = {nvirt, nvirt, nvirt, nvirt};
    int64_t eB[4] = {nvirt, nvirt, nocc, nocc};
    int64_t eD[4] = {nvirt, nvirt, nocc, nocc};
    int64_t sA[4], sB[4], sD[4];
    strides4(eA, sA); strides4(eB, sB); strides4(eD, sD);
    int64_t iA[4] = {'a', 'b', 'c', 'd'};
    int64_t iB[4] = {'c', 'd', 'i', 'j'};
    int64_t iD[4] = {'a', 'b', 'i', 'j'};

    TAPP_handle handle; TAPP_create_handle(&handle);
    TAPP_tensor_info A, B, C, D;
    TAPP_create_tensor_info(&A, handle, STORAGE, 4, eA, sA);
    TAPP_create_tensor_info(&B, handle, STORAGE, 4, eB, sB);
    TAPP_create_tensor_info(&C, handle, STORAGE, 4, eD, sD);
    TAPP_create_tensor_info(&D, handle, STORAGE, 4, eD, sD);

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, handle,
        TAPP_IDENTITY, A, iA, TAPP_IDENTITY, B, iB,
        TAPP_IDENTITY, C, iD, TAPP_IDENTITY, D, iD, PREC);

    TAPP_executor exec; TAPP_create_executor(&exec);
    TAPP_status status;

    int64_t nA = (int64_t)nvirt * nvirt * nvirt * nvirt;
    int64_t nB = (int64_t)nvirt * nvirt * nocc * nocc;
    int64_t nD = (int64_t)nvirt * nvirt * nocc * nocc;
    void *dA, *dB, *dC, *dD;
    cudaMalloc(&dA, nA * sizeof(Scalar));
    cudaMalloc(&dB, nB * sizeof(Scalar));
    cudaMalloc(&dC, nD * sizeof(Scalar));
    cudaMalloc(&dD, nD * sizeof(Scalar));
    cudaMemset(dA, 0x3f, nA * sizeof(Scalar));
    cudaMemset(dB, 0x3f, nB * sizeof(Scalar));
    cudaMemset(dC, 0x3f, nD * sizeof(Scalar));

    Scalar alpha = 0.5, beta = 1.0;

    TAPP_execute_product(plan, exec, &status, &alpha, dA, dB, &beta, dC, dD); // warm-up
    cudaDeviceSynchronize();

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        TAPP_execute_product(plan, exec, &status, &alpha, dA, dB, &beta, dC, dD);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    printf("PPL: %d iters, total %.3f ms, avg %.3f ms\n", ITERS, ms, ms / ITERS);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(A); TAPP_destroy_tensor_info(B);
    TAPP_destroy_tensor_info(C); TAPP_destroy_tensor_info(D);
    TAPP_destroy_executor(exec); TAPP_destroy_handle(handle);
}

// PHL: D_abij = A_kbcj * B_acik + C_abij  (needs transpose)
static void contraction_PHL(int nocc, int nvirt)
{
    int64_t eA[4] = {nocc, nvirt, nvirt, nocc};
    int64_t eB[4] = {nvirt, nvirt, nocc, nocc};
    int64_t eD[4] = {nvirt, nvirt, nocc, nocc};
    int64_t sA[4], sB[4], sD[4];
    strides4(eA, sA); strides4(eB, sB); strides4(eD, sD);
    int64_t iA[4] = {'k', 'b', 'c', 'j'};
    int64_t iB[4] = {'a', 'c', 'i', 'k'};
    int64_t iD[4] = {'a', 'b', 'i', 'j'};

    TAPP_handle handle; TAPP_create_handle(&handle);
    TAPP_tensor_info A, B, C, D;
    TAPP_create_tensor_info(&A, handle, STORAGE, 4, eA, sA);
    TAPP_create_tensor_info(&B, handle, STORAGE, 4, eB, sB);
    TAPP_create_tensor_info(&C, handle, STORAGE, 4, eD, sD);
    TAPP_create_tensor_info(&D, handle, STORAGE, 4, eD, sD);

    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, handle,
        TAPP_IDENTITY, A, iA, TAPP_IDENTITY, B, iB,
        TAPP_IDENTITY, C, iD, TAPP_IDENTITY, D, iD, PREC);

    TAPP_executor exec; TAPP_create_executor(&exec);
    TAPP_status status;

    int64_t nA = (int64_t)nocc * nvirt * nvirt * nocc;
    int64_t nB = (int64_t)nvirt * nvirt * nocc * nocc;
    int64_t nD = (int64_t)nvirt * nvirt * nocc * nocc;
    void *dA, *dB, *dC, *dD;
    cudaMalloc(&dA, nA * sizeof(Scalar));
    cudaMalloc(&dB, nB * sizeof(Scalar));
    cudaMalloc(&dC, nD * sizeof(Scalar));
    cudaMalloc(&dD, nD * sizeof(Scalar));
    cudaMemset(dA, 0x3f, nA * sizeof(Scalar));
    cudaMemset(dB, 0x3f, nB * sizeof(Scalar));
    cudaMemset(dC, 0x3f, nD * sizeof(Scalar));

    Scalar alpha = 1.0, beta = 1.0;

    TAPP_execute_product(plan, exec, &status, &alpha, dA, dB, &beta, dC, dD); // warm-up
    cudaDeviceSynchronize();

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        TAPP_execute_product(plan, exec, &status, &alpha, dA, dB, &beta, dC, dD);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    printf("PHL: %d iters, total %.3f ms, avg %.3f ms\n", ITERS, ms, ms / ITERS);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(A); TAPP_destroy_tensor_info(B);
    TAPP_destroy_tensor_info(C); TAPP_destroy_tensor_info(D);
    TAPP_destroy_executor(exec); TAPP_destroy_handle(handle);
}

int main(int argc, char const* argv[])
{
    int nocc = 20, nvirt = 200, nocc_PH = 100, nvirt_PH = 200;
    if (argc >= 5) {
        nocc = std::stoi(argv[1]); nvirt = std::stoi(argv[2]);
        nocc_PH = std::stoi(argv[3]); nvirt_PH = std::stoi(argv[4]);
    } else if (argc >= 3) {
        nocc = nocc_PH = std::stoi(argv[1]); nvirt = nvirt_PH = std::stoi(argv[2]);
    }
    printf("PPL nocc=%d nvirt=%d\n", nocc, nvirt);
    contraction_PPL(nocc, nvirt);
    printf("PHL nocc=%d nvirt=%d\n", nocc_PH, nvirt_PH);
    contraction_PHL(nocc_PH, nvirt_PH);
    return 0;
}
