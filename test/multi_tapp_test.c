/*
 * Test for multi-tapp: each command line argument is the path to a TAPP
 * implementation (shared library). The same contraction is run through every
 * backend and compared to the expected result. With two or more backends, it
 * also checks that objects from one backend are rejected by another.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "multi_tapp.h"

#define MAX_BACKENDS 8

static int failures = 0;

static void check(bool condition, const char* lib, const char* what)
{
    printf("%s %s: %s\n", condition ? "ok  " : "FAIL", lib, what);
    if (!condition) failures++;
}

struct objects
{
    Multi_TAPP_backend backend;
    TAPP_tensor_info info_A, info_B, info_C, info_D;
    TAPP_tensor_product plan;
    TAPP_executor exec;
};

// D_ij = alpha * A_ik * B_kj + beta * C_ij, with 2 x 2 column-major tensors
static bool create_objects(const char* lib, struct objects* o)
{
    if (Multi_TAPP_create_backend(&o->backend, lib) != 0)
    {
        check(false, lib, "create backend");
        return false;
    }
    check(Multi_TAPP_start(o->backend) == 0, lib, "start backend");
    check(Multi_TAPP_start(o->backend) == 0, lib, "start backend again");

    int64_t extents[2] = {2, 2};
    int64_t strides[2] = {1, 2};
    TAPP_error error = 0;
    error |= Multi_TAPP_create_tensor_info(o->backend, &o->info_A, TAPP_F64, 2, extents, strides);
    error |= Multi_TAPP_create_tensor_info(o->backend, &o->info_B, TAPP_F64, 2, extents, strides);
    error |= Multi_TAPP_create_tensor_info(o->backend, &o->info_C, TAPP_F64, 2, extents, strides);
    error |= Multi_TAPP_create_tensor_info(o->backend, &o->info_D, TAPP_F64, 2, extents, strides);
    check(error == 0, lib, "create tensor infos");

    int64_t read_extents[2] = {0, 0};
    int64_t read_strides[2] = {0, 0};
    Multi_TAPP_get_extents(o->backend, o->info_A, read_extents);
    Multi_TAPP_get_strides(o->backend, o->info_A, read_strides);
    check(Multi_TAPP_get_nmodes(o->backend, o->info_A) == 2 &&
          read_extents[0] == 2 && read_extents[1] == 2 &&
          read_strides[0] == 1 && read_strides[1] == 2,
          lib, "read back nmodes, extents and strides");

    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_D[2] = {'i', 'j'};
    error = Multi_TAPP_create_executor(o->backend, &o->exec, 0);
    check(error == 0, lib, "create executor with default attributes");

    error = Multi_TAPP_create_tensor_product(o->backend, &o->plan, o->exec,
                                             TAPP_IDENTITY, o->info_A, idx_A,
                                             TAPP_IDENTITY, o->info_B, idx_B,
                                             TAPP_IDENTITY, o->info_C, idx_D,
                                             TAPP_IDENTITY, o->info_D, idx_D,
                                             TAPP_DEFAULT_PREC);
    check(error == 0, lib, "create tensor product");

    return failures == 0;
}

static void run_contraction(const char* lib, struct objects* o)
{
    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4] = {1, 1, 1, 1};
    double D[4] = {0, 0, 0, 0};
    double alpha = 2;
    double beta = 3;
    double expected[4] = {49, 71, 65, 95};

    TAPP_status status = 0;
    TAPP_error error = Multi_TAPP_execute_product(o->backend, o->plan, o->exec, &status, &alpha, A, B, &beta, C, D);
    check(error == 0, lib, "execute product");
    check(Multi_TAPP_check_success(o->backend, error), lib, "check_success on the execute error code");

    // Waiting is optional: synchronous implementations may not provide it
    error = Multi_TAPP_wait_one(o->backend, o->exec, status);
    check(error == 0 || error == MULTI_TAPP_ERROR_UNIMPLEMENTED, lib, "wait_one");
    error = Multi_TAPP_wait_all(o->backend, o->exec);
    check(error == 0 || error == MULTI_TAPP_ERROR_UNIMPLEMENTED, lib, "wait_all");

    // Getting the error of a status is optional too
    error = Multi_TAPP_status_get_error(o->backend, status);
    check(error == 0 || error == MULTI_TAPP_ERROR_UNIMPLEMENTED, lib, "status_get_error after the product");
    check(Multi_TAPP_destroy_status(o->backend, status) == 0, lib, "destroy status");

    bool correct = true;
    for (int i = 0; i < 4; i++)
    {
        if (fabs(D[i] - expected[i]) > 1e-12) correct = false;
    }
    check(correct, lib, "contraction result");

    // Without a status: the caller does not want one, and the backend must get NULL to write it to
    double D_no_status[4] = {0, 0, 0, 0};
    error = Multi_TAPP_execute_product(o->backend, o->plan, o->exec, NULL, &alpha, A, B, &beta, C, D_no_status);
    check(error == 0, lib, "execute product without a status");
    error = Multi_TAPP_wait_all(o->backend, o->exec);
    check(error == 0 || error == MULTI_TAPP_ERROR_UNIMPLEMENTED, lib, "wait_all after the product without a status");
    correct = true;
    for (int i = 0; i < 4; i++)
    {
        if (fabs(D_no_status[i] - expected[i]) > 1e-12) correct = false;
    }
    check(correct, lib, "contraction result without a status");
}

// An error from the backend itself must not be mistaken for an error of multi-tapp
static void check_backend_error(const char* lib, struct objects* o)
{
    // Invalid: C and D do not have the same indices
    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_C[2] = {'i', 'x'};
    int64_t idx_D[2] = {'i', 'j'};
    TAPP_tensor_product plan;
    TAPP_error error = Multi_TAPP_create_tensor_product(o->backend, &plan, o->exec,
                                                        TAPP_IDENTITY, o->info_A, idx_A,
                                                        TAPP_IDENTITY, o->info_B, idx_B,
                                                        TAPP_IDENTITY, o->info_C, idx_C,
                                                        TAPP_IDENTITY, o->info_D, idx_D,
                                                        TAPP_DEFAULT_PREC);
    check(error > MULTI_TAPP_ERROR_LAST, lib, "backend error is outside the multi-tapp error codes");
    check(!Multi_TAPP_check_success(o->backend, error), lib, "check_success on a backend error code");

    char message[256];
    size_t length = Multi_TAPP_explain_error(o->backend, error, sizeof(message), message);
    check(length > 0 && length == strlen(message), lib, "explain_error on a backend error code");
    printf("     backend error %d (returned as %d): %s\n", error - MULTI_TAPP_ERROR_LAST, error, message);
}

// C_ji = alpha * A_ij + beta * B_ji, with 2 x 2 column-major tensors. A backend without permutes must report that
static void check_permute(const char* lib, struct objects* o)
{
    int64_t idx_ij[2] = {'i', 'j'};
    int64_t idx_ji[2] = {'j', 'i'};
    TAPP_tensor_permute plan;
    TAPP_error error = Multi_TAPP_create_tensor_permute(o->backend, &plan, o->exec,
                                                        TAPP_IDENTITY, o->info_A, idx_ij,
                                                        TAPP_IDENTITY, o->info_C, idx_ji,
                                                        TAPP_IDENTITY, o->info_D, idx_ji,
                                                        TAPP_DEFAULT_PREC);
    if (error == MULTI_TAPP_ERROR_UNIMPLEMENTED)
    {
        check(true, lib, "permute not provided, reported as unimplemented");
        return;
    }
    check(error == 0, lib, "create permute");
    if (error != 0) return;

    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4] = {0, 0, 0, 0};
    double alpha = 2;
    double beta = 3;
    TAPP_status status = 0;
    error = Multi_TAPP_execute_permute(o->backend, plan, o->exec, &status, &alpha, A, &beta, B, C);
    check(error == 0, lib, "execute permute");
    // Waiting is optional: synchronous implementations may not provide it
    error = Multi_TAPP_wait_all(o->backend, o->exec);
    check(error == 0 || error == MULTI_TAPP_ERROR_UNIMPLEMENTED, lib, "wait_all after the permute");
    check(Multi_TAPP_destroy_status(o->backend, status) == 0, lib, "destroy status of the permute");

    bool correct = true;
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            double expected = alpha * A[i + 2 * j] + beta * B[j + 2 * i];
            if (fabs(C[j + 2 * i] - expected) > 1e-12) correct = false;
        }
    }
    check(correct, lib, "permute result");
    check(Multi_TAPP_destroy_tensor_permute(o->backend, plan) == 0, lib, "destroy permute");
}

// Objects from "other" must be rejected when used with "o"'s backend
static void check_foreign_objects(const char* lib, struct objects* o, struct objects* other)
{
    TAPP_error error = Multi_TAPP_set_nmodes(o->backend, other->info_A, 2);
    check(error == MULTI_TAPP_ERROR_TENSOR_INFO_WRONG_BACKEND, lib, "reject tensor info from another backend");
    check(!Multi_TAPP_check_success(o->backend, error), lib, "check_success on a multi-tapp error code");

    int64_t idx_A[2] = {'i', 'k'};
    int64_t idx_B[2] = {'k', 'j'};
    int64_t idx_D[2] = {'i', 'j'};
    TAPP_tensor_product plan;
    error = Multi_TAPP_create_tensor_product(o->backend, &plan, o->exec,
                                             TAPP_IDENTITY, other->info_A, idx_A,
                                             TAPP_IDENTITY, o->info_B, idx_B,
                                             TAPP_IDENTITY, o->info_C, idx_D,
                                             TAPP_IDENTITY, o->info_D, idx_D,
                                             TAPP_DEFAULT_PREC);
    check(error == MULTI_TAPP_ERROR_TENSOR_INFO_A_WRONG_BACKEND, lib, "reject tensor product using tensor info from another backend");
    error = Multi_TAPP_create_tensor_product(o->backend, &plan, other->exec,
                                             TAPP_IDENTITY, o->info_A, idx_A,
                                             TAPP_IDENTITY, o->info_B, idx_B,
                                             TAPP_IDENTITY, o->info_C, idx_D,
                                             TAPP_IDENTITY, o->info_D, idx_D,
                                             TAPP_DEFAULT_PREC);
    check(error == MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND, lib, "reject tensor product using executor from another backend");

    double A[4] = {1, 2, 3, 4};
    double B[4] = {5, 6, 7, 8};
    double C[4] = {1, 1, 1, 1};
    double D[4] = {0, 0, 0, 0};
    double alpha = 2;
    double beta = 3;
    TAPP_status status = 0;
    error = Multi_TAPP_execute_product(o->backend, other->plan, o->exec, &status, &alpha, A, B, &beta, C, D);
    check(error == MULTI_TAPP_ERROR_TENSOR_PRODUCT_WRONG_BACKEND, lib, "reject tensor product from another backend");
    error = Multi_TAPP_execute_product(o->backend, o->plan, other->exec, &status, &alpha, A, B, &beta, C, D);
    check(error == MULTI_TAPP_ERROR_EXECUTOR_WRONG_BACKEND, lib, "reject executor from another backend");
    check(status == 0, lib, "no status from a rejected execution");

    // A status from the other backend
    error = Multi_TAPP_execute_product(other->backend, other->plan, other->exec, &status, &alpha, A, B, &beta, C, D);
    check(error == 0, lib, "execute product on the other backend");
    // A backend without wait_one reports that before it looks at the status
    error = Multi_TAPP_wait_one(o->backend, o->exec, status);
    check(error == MULTI_TAPP_ERROR_STATUS_WRONG_BACKEND || error == MULTI_TAPP_ERROR_UNIMPLEMENTED,
          lib, "reject status from another backend in wait_one");
    check(Multi_TAPP_destroy_status(o->backend, status) == MULTI_TAPP_ERROR_STATUS_WRONG_BACKEND,
          lib, "reject status from another backend in destroy_status");
    Multi_TAPP_wait_all(other->backend, other->exec);
    check(Multi_TAPP_destroy_status(other->backend, status) == 0, lib, "destroy status with its own backend");
}

// A library that can not be loaded gives an error that can be explained without a backend
static void check_missing_library(void)
{
    const char* lib = "missing library";
    Multi_TAPP_backend backend = 0;
    TAPP_error error = Multi_TAPP_create_backend(&backend, "./this-tapp-implementation-does-not-exist.so");
    check(error == MULTI_TAPP_ERROR_DLOPEN_FAILED, lib, "create backend fails");
    check(!Multi_TAPP_check_success(0, error), lib, "check_success without a backend");

    char message[64];
    size_t length = Multi_TAPP_explain_error(0, error, sizeof(message), message);
    check(length > 0 && length == strlen(message), lib, "explain_error without a backend");
    check(Multi_TAPP_explain_error(0, error, 0, NULL) == length, lib, "explain_error length query");

    // An error code of a backend can not be explained without the backend, but must not crash
    TAPP_error backend_error = MULTI_TAPP_ERROR_LAST + 1;
    length = Multi_TAPP_explain_error(0, backend_error, sizeof(message), message);
    check(length > 0 && length == strlen(message), lib, "explain_error on a backend error code without a backend");
    check(!Multi_TAPP_check_success(0, backend_error), lib, "check_success on a backend error code without a backend");
}

static void destroy_objects(const char* lib, struct objects* o)
{
    TAPP_error error = 0;
    error |= Multi_TAPP_destroy_tensor_product(o->backend, o->plan);
    error |= Multi_TAPP_destroy_executor(o->backend, o->exec);
    error |= Multi_TAPP_destroy_tensor_info(o->backend, o->info_A);
    error |= Multi_TAPP_destroy_tensor_info(o->backend, o->info_B);
    error |= Multi_TAPP_destroy_tensor_info(o->backend, o->info_C);
    error |= Multi_TAPP_destroy_tensor_info(o->backend, o->info_D);
    error |= Multi_TAPP_shutdown(o->backend);
    error |= Multi_TAPP_destroy_backend(o->backend);
    check(error == 0, lib, "destroy objects and backend");
}

int main(int argc, char** argv)
{
    int num_backends = argc - 1;
    if (num_backends < 1 || num_backends > MAX_BACKENDS)
    {
        fprintf(stderr, "usage: %s <tapp implementation library>...\n", argv[0]);
        return 2;
    }

    check_missing_library();

    struct objects objects[MAX_BACKENDS];
    for (int i = 0; i < num_backends; i++)
    {
        if (!create_objects(argv[i + 1], &objects[i]))
        {
            printf("%d failure(s)\n", failures);
            return 1;
        }
    }

    for (int i = 0; i < num_backends; i++)
    {
        run_contraction(argv[i + 1], &objects[i]);
        check_backend_error(argv[i + 1], &objects[i]);
        check_permute(argv[i + 1], &objects[i]);
    }

    for (int i = 0; i < num_backends; i++)
    {
        for (int j = 0; j < num_backends; j++)
        {
            if (i != j) check_foreign_objects(argv[i + 1], &objects[i], &objects[j]);
        }
    }

    for (int i = 0; i < num_backends; i++)
    {
        destroy_objects(argv[i + 1], &objects[i]);
    }

    printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
