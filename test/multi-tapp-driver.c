/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - Januari 2026
 */

#include <tapp.h>
#include "helpers.h"
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char const *argv[])
{
    TAPP_handle handle; // Declare handle
    TAPP_create_handle(0, &handle); // Choose implementation and create handle

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

    TAPP_create_tensor_info(&info_A, handle, TAPP_F32, nmode_A, extents_A, strides_A); // Assign the structure to the variable, including datatype

    // Tensor B
    int nmode_B = 4;
    int64_t extents_B[4] = {3, 2, 2, 3};
    int64_t strides_B[4] = {1, 3, 6, 12};
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, handle, TAPP_F32, nmode_B, extents_B, strides_B);

    // Tensor C
    int nmode_C = 3;
    int64_t extents_C[3] = {4, 2, 2};
    int64_t strides_C[3] = {1, 4, 8};
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, handle, TAPP_F32, nmode_C, extents_C, strides_C);

    // Output tensor D
    int nmode_D = 3;
    int64_t extents_D[3] = {4, 2, 2};
    int64_t strides_D[3] = {1, 4, 8};
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, handle, TAPP_F32, nmode_D, extents_D, strides_D);

    /*
     * Decide how the calculation should be executed, which indices to contract, elemental operations and precision.
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

    TAPP_tensor_product plan; // Declare the variable that holds the information about the calculation 

    TAPP_create_tensor_product(&plan, handle, op_A, info_A, idx_A, op_B, info_B, idx_B, op_C, info_C, idx_C, op_D, info_D, idx_D, prec); // Assign the calculation options to the variable

    /*
     * Decide create executor (not yet implemented)
     */

    TAPP_executor exec; // Declaration of executor
    TAPP_create_executor(&exec, handle); // Creation of executor
    // int exec_id = 1; // Choose executor
    // exec = (intptr_t)&exec_id; // Assign executor

    /*
     * Status objects are used to know the status of the execution process (not yet implemented)
     */

    TAPP_status status; // Declare status object

    /*
     * Choose data for the execution
     */

    float alpha = 1; // Choose the scalar for scaling A * B

    float A[36] = { // Choose data in tensor A
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1,

        1, 2, 1.01, -1,
        1, 2, 1.01, -1,
        1, 2, 1.01, -1};

    float B[36] = { // Choose data for tensor B
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

    float beta = 0; // Choose scalar for scaling C

    float C[16] = { // Choose data for tensor C
        2, 4, 6, 8,
        2, 4, 6, 8,

        2, 4, 6, 8,
        2, 4, 6, 8};

    float D[16] = { // Allocate tensor D (will be overwritten by the process)
        1, 2, 3, 4,
        5, 6, 7, 8,

        1, 2, 3, 4,
        5, 6, 7, 8};

    /*
     * Execution 
     */

    TAPP_error error = TAPP_execute_product(plan, handle, exec, &status, (void *)&alpha, (void *)A, (void *)B, (void *)&beta, (void *)C, (void *)D); // Execute the product with a plan, executor, status object and data, returning an error object

    /*
     * Error handling
     */

    printf(TAPP_check_success(error, handle) ? "Success\n" : "Fail\n"); // Print whether or not the operation was successful
    int message_len = TAPP_explain_error(error, handle, 0, NULL); // Get size of error message
    char *message_buff = malloc((message_len + 1) * sizeof(char)); // Allocate buffer for message, including null terminator
    TAPP_explain_error(error, handle, message_len + 1, message_buff); // Fetch error message
    printf("%s", message_buff); // Print message
    free(message_buff); // Free buffer
    printf("\n");

    print_tensor_s(nmode_D, extents_D, strides_D, D); // Print tensor

    /*
     * Free structures
     */

    TAPP_destroy_tensor_product(plan, handle);
    TAPP_destroy_tensor_info(info_A, handle);
    TAPP_destroy_tensor_info(info_B, handle);
    TAPP_destroy_tensor_info(info_C, handle);
    TAPP_destroy_tensor_info(info_D, handle);
    TAPP_destroy_executor(exec, handle);
    TAPP_destroy_handle(handle);

    return 0;
}