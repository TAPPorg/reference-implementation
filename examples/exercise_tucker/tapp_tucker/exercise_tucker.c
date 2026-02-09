#include "exercise_tucker.h"

/*
 * TODO:
 *  1. Complete the function call for TAPP_execute_product (Line 102).
 *  2. Complete the function call for TAPP_create_tensor_product (Line 67)
 *  3. Complete the function call for TAPP_create_tensor_info (Line 32)
 */

void* tucker_to_tensor_contraction(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A,
                                   int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B,
                                   int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D,
                                   int64_t* idx_A, int64_t* idx_B, int64_t* idx_D)
{
    TAPP_handle handle; // Declare handle
    TAPP_create_handle(&handle); // Create handle

    /*
     * The tensor product looks in a simplified way as follows: D <- a*A*B+b*C.
     * Where the lowercase letters are constants and uppercase are tensors.
     * The operation requires four tensors that all needs to be initialized.
     */

    // Initialize the structures of the tensors

    // Tensor A

    TAPP_tensor_info info_A; // Declare the variable that holds the tensor structure

    /*
     * TODO 3: Complete the function call.
     * Uncomment function call
     * Add: nmode_A, extents_A, and strides_A
     */
    //TAPP_create_tensor_info(&info_A, handle, TAPP_F64, , , ); // Assign the structure to the variable, including datatype

    // Tensor B
    TAPP_tensor_info info_B;
    TAPP_create_tensor_info(&info_B, handle, TAPP_F64, nmode_B, extents_B, strides_B);

    // Tensor C
    TAPP_tensor_info info_C;
    TAPP_create_tensor_info(&info_C, handle, TAPP_F64, nmode_D, extents_D, strides_D);

    // Output tensor D
    TAPP_tensor_info info_D;
    TAPP_create_tensor_info(&info_D, handle, TAPP_F64, nmode_D, extents_D, strides_D);

    /*
     * Decide how the calculation should be executed, which indices to contract, elemental operations and precision.
     */

    // Decide elemental operations (conjugate available for complex datatypes)
    TAPP_element_op op_A = TAPP_IDENTITY; // Decide elemental operation for tensor A
    TAPP_element_op op_B = TAPP_IDENTITY; // Decide elemental operation for tensor B
    TAPP_element_op op_C = TAPP_IDENTITY; // Decide elemental operation for tensor C
    TAPP_element_op op_D = TAPP_IDENTITY; // Decide elemental operation for tensor D

    TAPP_prectype prec = TAPP_DEFAULT_PREC; //Choose the calculation precision

    TAPP_tensor_product plan; // Declare the variable that holds the information about the calculation 

    /*
     * TODO 2: Complete the function call.
     * Uncomment function call
     * Add: idx_A, idx_B, and ixd_D
     */
    //TAPP_create_tensor_product(&plan, handle, op_A, info_A, , op_B, info_B, , op_C, info_C, idx_D, op_D, info_D, , prec); // Assign the calculation options to the variable

    /*
     * Decide which implementation to use with a executor (not yet implemented)
     */

    TAPP_executor exec; // Declaration of executor
    TAPP_create_executor(&exec); // Creation of executor
    // int exec_id = 1; // Choose executor
    // exec = (intptr_t)&exec_id; // Assign executor

    /*
     * Status objects are used to know the status of the execution process (not yet implemented)
     */

    TAPP_status status; // Declare status object

    /*
     * Choose data for the execution
     */

    double alpha = 1; // Choose the scalar for scaling A * B

    double beta = 0; // Choose scalar for scaling C

    /*
     * Execution 
     */

    TAPP_error error;
    /*
     * TODO 1: Complete the function call.
     * Uncomment function call
     * Add: A, B, and D
     */
    //error = TAPP_execute_product(plan, exec, &status, (void *)&alpha, , , (void *)&beta, (void *)NULL, ); // Execute the product with a plan, executor, status object and data, returning an error object
    /*
     * Error handling
     */

    if (!TAPP_check_success(error)) {
        int message_len = TAPP_explain_error(error, 0, NULL); // Get size of error message
        char *message_buff = malloc((message_len + 1) * sizeof(char)); // Allocate buffer for message, including null terminator
        TAPP_explain_error(error, message_len + 1, message_buff); // Fetch error message
        printf("%s", message_buff); // Print message
        free(message_buff); // Free buffer
        printf("\n");
    }

    /*
     * Free structures
     */

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_tensor_info(info_A);
    TAPP_destroy_tensor_info(info_B);
    TAPP_destroy_tensor_info(info_D);
    TAPP_destroy_executor(exec);

    return D;
}