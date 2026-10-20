/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - September 2025
 */

#include "tapp_cutensor.h"
#include <tapp.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(int argc, char const *argv[])
{
    // Start TAPP before any other TAPP function is called
    TAPP_start();

    /*
     * The tensor product looks in a simplified way as follows: D <- a*A*B+b*C.
     * Where the lowercase letters are constants and uppercase are tensors.
     * The operation requires four tensors that all needs to be initialized.
     */

    // Initialize the structures of the tensors

    // Tensor A

    int nmode_A = 3; // Decide the number of indices/order/rank for the tensor

    int64_t extents_A[3] = {4, 3, 3}; // Decide the shape of the tensor/extents of the dimensions

    int64_t strides_A[3] = {1, 4, 12}; // Decide the memory structure of the tensor/memory jumps for each dimension

    TAPP_tensor_info info_A; // Declare the variable that holds the tensor structure

    TAPP_create_tensor_info(&info_A, TAPP_F64, nmode_A, extents_A, strides_A); // Assign the structure to the variable, including datatype

    // Tensor B
    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, TAPP_F64, nmode_B, extents_B, strides_B);

    // Tensor C
    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, TAPP_F64, nmode_C, extents_C, strides_C);

    // Output tensor D
    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, TAPP_F64, nmode_D, extents_D, strides_D);

    /*
     * Decide who the calculation should be executed, which indices to contract, elemental operations and precision.
     */

    // Decide elemental operations (conjugate available for complex datatypes)
    TAPP_element_op op_A = TAPP_IDENTITY; // Decide elemental operation for tensor A
    TAPP_element_op op_B = TAPP_IDENTITY; // Decide elemental operation for tensor B
    TAPP_element_op op_C = TAPP_IDENTITY; // Decide elemental operation for tensor C
    TAPP_element_op op_D = TAPP_IDENTITY; // Decide elemental operation for tensor D

    /*
     * Decide which indices to contract
     * Tensor C and D should contain the same indices
     * Indices not in D are contracted
     * Indices in A, B and D are calculated similar to an hadamard product
     */
    int64_t idx_A[3] = {'a', 'b', 'c'}; // Decide indices for tensor A
    int64_t idx_B[4] = {'c', 'd', 'e', 'b'}; // Decide indices for tensor B
    int64_t idx_C[3] = {'a', 'd', 'e'}; // Decide indices for tensor C, should contain the same as D
    int64_t idx_D[3] = {'a', 'd', 'e'}; // Decide indices for the output/tensor D

    TAPP_prectype prec = TAPP_DEFAULT_PREC; //Choose the calculation precision

    /*
     * Decide which device and resources to use with a executor
     */

    TAPP_attr attr;
    TAPP_create_attr(&attr);

    int device = 0;
    TAPP_attr_set(attr, TAPP_CUTENSOR_DEVICE, sizeof(int), &device); // Choose the GPU

    // Choose the stream the executor runs on. It must be created on the chosen device and stays owned by us: destroy
    // it after the executor. Without TAPP_CUTENSOR_STREAM the executor creates a stream of its own.
    cudaSetDevice(device);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    TAPP_attr_set(attr, TAPP_CUTENSOR_STREAM, sizeof(cudaStream_t), &stream);

    TAPP_executor exec; // Declaration of executor
    TAPP_create_executor(&exec, attr); // Creation of executor, which copies the attributes

    TAPP_destroy_attr(attr); // The executor has copied the attributes, so they can be destroyed

    TAPP_tensor_product plan; // Declare the variable that holds the information about the calculation

    TAPP_create_tensor_product(&plan, exec, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec); // Assign the calculation options to the variable

    /*
     * Status objects are used to know the status of the execution process
     */

    TAPP_status status; // Declare status object

    /*
     * Choose data for the execution
     */

    double alpha = 1; // Choose the scalar for scaling A * B

    double A[36] = { // Choose data in tensor A
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    double* dA;
    cudaMalloc((void**)&dA, 36 * sizeof(double));
    cudaMemcpy(dA, A, 36 * sizeof(double), cudaMemcpyHostToDevice);

    double B[36] = { // Choose data for tensor B
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

    double* dB;
    cudaMalloc((void**)&dB, 36 * sizeof(double));
    cudaMemcpy(dB, B, 36 * sizeof(double), cudaMemcpyHostToDevice);

    double beta = 0; // Choose scalar for scaling C

    double C[16] = { // Choose data for tensor C
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    double* dC;
    cudaMalloc((void**)&dC, 16 * sizeof(double));
    cudaMemcpy(dC, C, 16 * sizeof(double), cudaMemcpyHostToDevice);

    double D[16] = { // Allocate tensor D (will be overwritten by the process)
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    double* dD;
    cudaMalloc((void**)&dD, 16 * sizeof(double));
    cudaMemcpy(dD, D, 16 * sizeof(double), cudaMemcpyHostToDevice);

    /*
     * Execution 
     */

    TAPP_error error = TAPP_execute_product(plan, exec, &status, (void *)&alpha, (void *)dA, (void *)dB, (void *)&beta, (void *)dC, (void *)dD); // Execute the product with a plan, executor, status object and data in device memory, returning an error object

    TAPP_wait_one(exec, status); // The product runs asynchronously on the stream, wait for it to finish
    // Could use TAPP_wait_all(exec);
    
    cudaMemcpy(D, dD, 16 * sizeof(double), cudaMemcpyDeviceToHost); // Copy the result back to the host

    /*
     * Error handling
     */

    printf(TAPP_check_success(error) ? "Success\n" : "Fail\n"); // Print whether or not the operation was successful
    int message_len = TAPP_explain_error(error, 0, NULL); // Get size of error message
    char *message_buff = malloc((message_len + 1) * sizeof(char)); // Allocate buffer for message, including null terminator
    TAPP_explain_error(error, message_len + 1, message_buff); // Fetch error message
    printf("%s", message_buff); // Print message
    free(message_buff); // Free buffer
    printf("\n");

    /*
     * Free cuda memory
     */

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);

    /*
     * Free structures
     */
    
    TAPP_destroy_status(status);
    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_C);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);
    cudaStreamDestroy(stream); // The stream was given through the attributes, so it is not destroyed with the executor

    // Shut down TAPP after the last use of it
    TAPP_shutdown();

    return 0;
}