/*
 * Jan Brandejs
 * IRIT Toulouse - May 2026
 */

#include <tapp.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <cassert>

// Added missing headers
#include <iostream>
#include <vector>
#include <cstdint>

#include <cuComplex.h>
#include <curand_kernel.h>

extern "C" {
    #include "helpers.h"
}


void contraction_PPL(const int nocc, const int nvirt);
void contraction_PHL(const int nocc, const int nvirt);
void print_tensor_c_cpp(int nmode, const int64_t *extents, const int64_t *strides, const std::complex<double> *data);




// Initialize random states
__global__ void init_curand(curandState *states, int64_t seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &states[id]);
}

// Generate Standard Normal (mean 0, std 1) with imaginary = 0.0
__global__ void gen_complex_gauss_kernel(cuDoubleComplex *out, int64_t N, curandState *states, double mean, double stddev) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    curandState localState = states[id];

    for (size_t i = id; i < N; i += stride) {
        double val = curand_normal_double(&localState) * stddev + mean;
        out[i] = make_cuDoubleComplex(val, 0.0);
    }
    states[id] = localState;
}

// Callable Wrapper
void generate_gaussian_tensor(cuDoubleComplex* d_out, int64_t N, curandState* d_states, double mean, double stddev) {
    int threads = 256;
    int blocks = 1024;
    gen_complex_gauss_kernel<<<blocks, threads>>>(d_out, N, d_states, mean, stddev);
}



int main(int argc, char const *argv[])
{
    int nocc = 20; // Default value
    int nvirt = 200; 
    int nocc_PH = 100; 
    int nvirt_PH = 200; 

    try {
        if (argc >= 5) {
            nocc = std::stoi(argv[1]);
            nvirt = std::stoi(argv[2]);
            nocc_PH = std::stoi(argv[3]);
            nvirt_PH = std::stoi(argv[4]);
        }
        else if (argc >= 3) {
            nocc = std::stoi(argv[1]);
            nvirt = std::stoi(argv[2]);
            nocc_PH = nocc;
            nvirt_PH = nvirt;
        }
    } catch (...) {
        std::cout << "Invalid input. Using defaults.\n";
    }

    std::cout << "double complex datatype \n";
    
    // Run the benchmark
    std::cout << "running contraction particle-particle ladder R_abij += 0.5 * V_abcd * t_cdij \n";
    std::cout << "PP with test size: nocc: " << nocc << " nvirt: " << nvirt << "\n";
    contraction_PPL(nocc, nvirt);
    std::cout << "running contraction particle-hole ladder R_abij += V_kbcj * t_acik \n";
    std::cout << "PH with test size: nocc: " << nocc_PH << " nvirt: " << nvirt_PH << "\n";
    contraction_PHL(nocc_PH, nvirt_PH);
    
    return 0;
}


// contraction particle-particle ladder R_abij += 0.5 * V_abcd * t_cdij
// i.e. D_abij = 0.5 * A_abcd * B_cdij + C_abij
// no transpose needed
void contraction_PPL(const int nocc, const int nvirt)
{
    TAPP_handle handle;
    TAPP_create_handle(&handle);

    int nmode_A = 4;
    int64_t extents_A[4] = {nvirt, nvirt, nvirt, nvirt};
    int64_t strides_A[4] = {1, nvirt, static_cast<int64_t>(nvirt)*nvirt, static_cast<int64_t>(nvirt)*nvirt*nvirt};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, handle, TAPP_C64, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {nvirt, nvirt, nocc, nocc};
    int64_t strides_B[4] = {1, nvirt, static_cast<int64_t>(nvirt)*nvirt, static_cast<int64_t>(nvirt)*nvirt*nocc};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, handle, TAPP_C64, nmode_B, extents_B, strides_B);

    int nmode_C = 4;
    int64_t extents_C[4] = {nvirt, nvirt, nocc, nocc};
    int64_t strides_C[4] = {1, nvirt, static_cast<int64_t>(nvirt)*nvirt, static_cast<int64_t>(nvirt)*nvirt*nocc};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, handle, TAPP_C64, nmode_C, extents_C, strides_C);

    int nmode_D = 4;
    int64_t extents_D[4] = {nvirt, nvirt, nocc, nocc};
    int64_t strides_D[4] = {1, nvirt, static_cast<int64_t>(nvirt)*nvirt, static_cast<int64_t>(nvirt)*nvirt*nocc};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, handle, TAPP_C64, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;

    int64_t idx_A[4] = {'a', 'b', 'c', 'd'};
    int64_t idx_B[4] = {'c', 'd', 'i', 'j'};
    int64_t idx_C[4] = {'a', 'b', 'i', 'j'};
    int64_t idx_D[4] = {'a', 'b', 'i', 'j'};

    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    TAPP_create_executor(&exec);
    TAPP_status status;

    int64_t A_nelem = static_cast<int64_t>(nvirt)*nvirt*nvirt*nvirt;
    int64_t B_nelem = static_cast<int64_t>(nvirt)*nvirt*nocc*nocc;
    int64_t C_nelem = static_cast<int64_t>(nvirt)*nvirt*nocc*nocc;
    int64_t D_nelem = static_cast<int64_t>(nvirt)*nvirt*nocc*nocc;

    std::complex<double> alpha = 0.5;
    std::complex<double> beta = 1;

    // Use std::vector to fix Variable Length Array (VLA) errors
    std::vector<std::complex<double>> A(A_nelem);
    std::vector<std::complex<double>> B(B_nelem);
    std::vector<std::complex<double>> C(C_nelem);
    std::vector<std::complex<double>> D(D_nelem);

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, A_nelem * sizeof(std::complex<double>));
    cudaMalloc((void**)&B_d, B_nelem * sizeof(std::complex<double>));
    cudaMalloc((void**)&C_d, C_nelem * sizeof(std::complex<double>));
    cudaMalloc((void**)&D_d, D_nelem * sizeof(std::complex<double>));

    int threads = 256, blocks = 1024; 
    curandState *d_states;
    cudaMalloc(&d_states, blocks * threads * sizeof(curandState));
    init_curand<<<blocks, threads>>>(d_states, 1234ULL);
    cudaDeviceSynchronize(); // Wait for init to finish

    // Cast as cuDoubleComplex* to match function signature
    generate_gaussian_tensor((cuDoubleComplex*)A_d, A_nelem, d_states, 0.01, 0.1); 
    generate_gaussian_tensor((cuDoubleComplex*)B_d, B_nelem, d_states, 0.01, 0.1); 
    generate_gaussian_tensor((cuDoubleComplex*)C_d, C_nelem, d_states, 0.01, 0.1); 

    // cast to uintptr_t
    assert(reinterpret_cast<uintptr_t>(A_d) % 128 == 0);
    assert(reinterpret_cast<uintptr_t>(B_d) % 128 == 0);
    assert(reinterpret_cast<uintptr_t>(C_d) % 128 == 0);
    assert(reinterpret_cast<uintptr_t>(D_d) % 128 == 0);
   
    // WARM-UP 
    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    cudaDeviceSynchronize(); // Wait for warm-up to finish

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // RECORD START
    cudaEventRecord(start);
    
    int iterations = 10;
    for (int i=0; i<10; i++){
       TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    // Pass .data() to get the underlying vector pointer
    cudaMemcpy((void*)D.data(), (void*)D_d, D_nelem * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

    //print_tensor_c_cpp(nmode_D, extents_D, strides_D, D.data());
    
    // CALCULATE ELAPSED TIME
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "PPL contr.: Total time for " << iterations << " iterations: " << milliseconds << " ms\n";
    std::cout << "PPL contr.: Average time per iteration: " << milliseconds / iterations << " ms\n";
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    if (d_states) cudaFree(d_states);

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
}


// contraction particle-hole ladder R_abij += V_kbcj * t_acik
// i.e. D_abij = A_kbcj * B_acik + C_abij
// this is a case with the transpose
void contraction_PHL(const int nocc, const int nvirt)
{
    TAPP_handle handle;
    TAPP_create_handle(&handle);

    int nmode_A = 4;
    int64_t extents_A[4] = {nocc, nvirt, nvirt, nocc};
    int64_t strides_A[4] = {1, nocc, static_cast<int64_t>(nocc)*nvirt, static_cast<int64_t>(nocc)*nvirt*nvirt};
    TAPP_tensor_info info_A;
    TAPP_create_tensor_info(&info_A, handle, TAPP_C64, nmode_A, extents_A, strides_A);

    int nmode_B = 4;
    int64_t extents_B[4] = {nvirt, nvirt, nocc, nocc};
    int64_t strides_B[4] = {1, nvirt, nvirt*nvirt, nvirt*nvirt*nocc};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, handle, TAPP_C64, nmode_B, extents_B, strides_B);

    int nmode_C = 4;
    int64_t extents_C[4] = {nvirt, nvirt, nocc, nocc};
    int64_t strides_C[4] = {1, nvirt, nvirt*nvirt, nvirt*nvirt*nocc};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, handle, TAPP_C64, nmode_C, extents_C, strides_C);

    int nmode_D = 4;
    int64_t extents_D[4] = {nvirt, nvirt, nocc, nocc};
    int64_t strides_D[4] = {1, nvirt, nvirt*nvirt, nvirt*nvirt*nocc};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, handle, TAPP_C64, nmode_D, extents_D, strides_D);

    TAPP_tensor_product plan;
    TAPP_element_op op_A = TAPP_IDENTITY;
    TAPP_element_op op_B = TAPP_IDENTITY;
    TAPP_element_op op_C = TAPP_IDENTITY;
    TAPP_element_op op_D = TAPP_IDENTITY;

    int64_t idx_A[4] = {'k', 'b', 'c', 'j'};
    int64_t idx_B[4] = {'a', 'c', 'i', 'k'};
    int64_t idx_C[4] = {'a', 'b', 'i', 'j'};
    int64_t idx_D[4] = {'a', 'b', 'i', 'j'};

    TAPP_prectype prec = TAPP_DEFAULT_PREC;
    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec);

    TAPP_executor exec;
    TAPP_create_executor(&exec);
    TAPP_status status;

    int64_t A_nelem = static_cast<int64_t>(nocc)*nvirt*nvirt*nocc;
    int64_t B_nelem = static_cast<int64_t>(nvirt)*nvirt*nocc*nocc;
    int64_t C_nelem = static_cast<int64_t>(nvirt)*nvirt*nocc*nocc;
    int64_t D_nelem = static_cast<int64_t>(nvirt)*nvirt*nocc*nocc;

    std::complex<double> alpha = 1;
    std::complex<double> beta = 1;

    // Use std::vector to fix Variable Length Array (VLA) errors
    std::vector<std::complex<double>> A(A_nelem);
    std::vector<std::complex<double>> B(B_nelem);
    std::vector<std::complex<double>> C(C_nelem);
    std::vector<std::complex<double>> D(D_nelem);

    void *A_d, *B_d, *C_d, *D_d; // Device pointers
    cudaMalloc((void**)&A_d, A_nelem * sizeof(std::complex<double>));
    cudaMalloc((void**)&B_d, B_nelem * sizeof(std::complex<double>));
    cudaMalloc((void**)&C_d, C_nelem * sizeof(std::complex<double>));
    cudaMalloc((void**)&D_d, D_nelem * sizeof(std::complex<double>));

    int threads = 256, blocks = 1024; 
    curandState *d_states;
    cudaMalloc(&d_states, blocks * threads * sizeof(curandState));
    init_curand<<<blocks, threads>>>(d_states, 1234ULL);
    cudaDeviceSynchronize(); // Wait for init to finish

    // Cast as cuDoubleComplex* to match function signature
    generate_gaussian_tensor((cuDoubleComplex*)A_d, A_nelem, d_states, 0.01, 0.1); 
    generate_gaussian_tensor((cuDoubleComplex*)B_d, B_nelem, d_states, 0.01, 0.1); 
    generate_gaussian_tensor((cuDoubleComplex*)C_d, C_nelem, d_states, 0.01, 0.1); 

    // ast to uintptr_t
    assert(reinterpret_cast<uintptr_t>(A_d) % 128 == 0);
    assert(reinterpret_cast<uintptr_t>(B_d) % 128 == 0);
    assert(reinterpret_cast<uintptr_t>(C_d) % 128 == 0);
    assert(reinterpret_cast<uintptr_t>(D_d) % 128 == 0);
   
    // WARM-UP 
    TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    cudaDeviceSynchronize(); // Wait for warm-up to finish

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // RECORD START
    cudaEventRecord(start);
    
    int iterations = 10;
    for (int i=0; i<10; i++){
       TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)A_d, (void *)B_d, (void *)&beta, (void *)C_d, (void *)D_d);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    // Pass .data() to get the underlying vector pointer
    cudaMemcpy((void*)D.data(), (void*)D_d, D_nelem * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

    //print_tensor_c_cpp(nmode_D, extents_D, strides_D, D.data());
    
    // CALCULATE ELAPSED TIME
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "PHL contr.: Total time for " << iterations << " iterations: " << milliseconds << " ms\n";
    std::cout << "PHL contr.: Average time per iteration: " << milliseconds / iterations << " ms\n";
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (C_d) cudaFree(C_d);
    if (D_d) cudaFree(D_d);
    if (d_states) cudaFree(d_states);

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
    TAPP_destroy_handle(handle);
}

void print_tensor_c_cpp(int nmode, const int64_t *extents, const int64_t *strides, const std::complex<double> *data)
{
    int64_t *coords = (int64_t *)malloc(nmode * sizeof(int64_t));
    int64_t size = 1;
    for (size_t i = 0; i < nmode; i++)
    {
        coords[i] = 0;
        size *= extents[i];
    }
    printf("\t");
    for (size_t j = 0; j < size; j++)
    {
        int64_t index = 0;
        for (size_t i = 0; i < nmode; i++)
        {
            index += coords[i] * strides[i];
        }
        printf("%.3f+%.3fi", data[index].real(), data[index].imag());

        if (nmode <= 0)
            continue;

        int k = 0;
        do
        {
            if (k != 0)
            {
                printf("\n");
                if (j < size - 1)
                {
                    printf("\t");
                }
            }
            else
            {
                printf(" ");
            }
            coords[k] = (coords[k] + 1) % extents[k];
            k++;
        } while (coords[k - 1] == 0 && k < nmode);
    }
    free(coords);
}
