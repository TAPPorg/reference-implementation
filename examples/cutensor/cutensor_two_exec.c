#include "tapp_cutensor.h"
#include <tapp.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * The asynchronous example of async.c on cuTENSOR: each executor has a CUDA stream of its own, and the tensors are in
 * device memory. Check that the products of the two executors overlap with
 * nsys profile ./tapp-cutensor-two-exec
 */

#define WARM_UP_RUNS 5
#define MEASURED_RUNS 5
#define N 1000 // extent of every mode: each product is about 2 * N^3 = 2 GFLOP

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t cerr_ = (call);                                                            \
        if (cerr_ != cudaSuccess) {                                                            \
            fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(cerr_)); \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

// D_ij = alpha * A_ik * B_kj + beta * C_ij, every tensor N x N in column-major order
int64_t ext[2] = {N, N};
int64_t strd[2] = {1, N};
int64_t idxa[2] = {'i', 'k'};
int64_t idxb[2] = {'k', 'j'};
int64_t idxd[2] = {'i', 'j'};

double alpha = 1.0;
double beta = 0.5;

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Fills data with pseudorandom numbers in [-1, 1], from the sequence seeded with srand
static void generate_data(double* data, size_t size)
{
    for (size_t i = 0; i < size; i++) data[i] = 2.0 * rand() / RAND_MAX - 1.0;
}

// Checks that two tensors with size elements in the same layout hold the same data. Implementations sum in different
// orders, so the elements only have to agree up to the relative tolerance (absolute for elements smaller than 1).
// Prints the first element that differs.
static bool same_data(const double* x, const double* y, size_t size, double tolerance)
{
    for (size_t i = 0; i < size; i++)
    {
        double diff = x[i] > y[i] ? x[i] - y[i] : y[i] - x[i];
        double abs_x = x[i] < 0 ? -x[i] : x[i];
        double abs_y = y[i] < 0 ? -y[i] : y[i];
        double scale = abs_x > abs_y ? abs_x : abs_y;
        if (scale < 1.0) scale = 1.0;
        if (!(diff <= tolerance * scale)) // also catches NaN
        {
            fprintf(stderr, "Element %zu differs: %.17g and %.17g\n", i, x[i], y[i]);
            return false;
        }
    }
    return true;
}

// Times the product on cuTENSOR with one and two executors, writing into dd1 and dd2, and checks the results against
// ddr. da, db, dc, dd1 and dd2 are in device memory, ddr and dd_host in host memory. A, B and C are only read, so both
// executors share them. Each executor writes its own D.
void run_example(const double* da, const double* db, const double* dc, const double* ddr, double* dd1, double* dd2, double* dd_host, const size_t size)
{
    TAPP_start();

    // Both executors run on device 0. Neither gets TAPP_CUTENSOR_STREAM, so each creates a stream of its own and the
    // products of the two can run concurrently. The executors copy the attributes, so they can be destroyed right away.
    TAPP_attr a;
    TAPP_create_attr(&a);
    int device = 0;
    TAPP_attr_set(a, TAPP_CUTENSOR_DEVICE, sizeof(int), &device);

    TAPP_executor et1;
    TAPP_create_executor(&et1, a);
    TAPP_executor et2;
    TAPP_create_executor(&et2, a);

    TAPP_destroy_attr(a);

    TAPP_tensor_info ta;
    TAPP_create_tensor_info(&ta, TAPP_DOUBLE, 2, ext, strd);
    TAPP_tensor_info tb;
    TAPP_create_tensor_info(&tb, TAPP_DOUBLE, 2, ext, strd);
    TAPP_tensor_info td;
    TAPP_create_tensor_info(&td, TAPP_DOUBLE, 2, ext, strd);

    // The plan is created with e1 but can be executed on any executor of the backend
    TAPP_tensor_product p;
    TAPP_create_tensor_product(&p, et1, TAPP_IDENTITY, ta, idxa, TAPP_IDENTITY, tb, idxb, TAPP_IDENTITY, td, idxd, TAPP_IDENTITY, td, idxd, TAPP_DEFAULT_PREC);

    TAPP_destroy_tensor_info(ta);
    TAPP_destroy_tensor_info(tb);
    TAPP_destroy_tensor_info(td);

    TAPP_status s1;
    TAPP_status s2;

    // Warmup
    for (size_t i = 0; i < WARM_UP_RUNS; i++) {
        TAPP_execute_product(p, et1, NULL, &alpha, da, db, &beta, dc, dd1);
        TAPP_execute_product(p, et2, NULL, &alpha, da, db, &beta, dc, dd2);
    }
    TAPP_wait_all(et1);
    TAPP_wait_all(et2);

    double tsmin = 0;
    double tsmean = 0;
    double twmin = 0;
    double twmean = 0;
    double ttmin = 0;
    double ttmean = 0;

    for (size_t i = 0; i < MEASURED_RUNS; i++) {
        double t0 = now();
        TAPP_execute_product(p, et1, &s1, &alpha, da, db, &beta, dc, dd1);
        TAPP_execute_product(p, et1, &s2, &alpha, da, db, &beta, dc, dd2);
        double t1 = now();   // the calls have returned, the products may still be running

        TAPP_wait_all(et1);
        double t2 = now();   // all products are done
        double ts = t1 - t0;
        double tw = t2 - t1;
        double tt = t2 - t0;

        tsmin = tsmin == 0.0 || tsmin > ts ? ts : tsmin;
        tsmean += ts;

        twmin = twmin == 0.0 || twmin > tw ? tw : twmin;
        twmean += tw;

        ttmin = ttmin == 0.0 || ttmin > tt ? tt : ttmin;
        ttmean += tt;

        TAPP_destroy_status(s1);
        TAPP_destroy_status(s2);
    }

    tsmean /= MEASURED_RUNS;
    twmean /= MEASURED_RUNS;
    ttmean /= MEASURED_RUNS;

    printf("Single executor:\n");
    printf("\tsubmit: min %.3f ms, mean %.3f ms\n", tsmin * 1e3, tsmean * 1e3);
    printf("\twait:   min %.3f ms, mean %.3f ms\n", twmin * 1e3, twmean * 1e3);
    printf("\ttotal:  min %.3f ms, mean %.3f ms\n", ttmin * 1e3, ttmean * 1e3);
    printf("\n");

    tsmin = 0;
    tsmean = 0;
    twmin = 0;
    twmean = 0;
    ttmin = 0;
    ttmean = 0;

    for (size_t i = 0; i < MEASURED_RUNS; i++) {
        double t0 = now();
        TAPP_execute_product(p, et1, &s1, &alpha, da, db, &beta, dc, dd1);
        TAPP_execute_product(p, et2, &s2, &alpha, da, db, &beta, dc, dd2);
        double t1 = now();   // the calls have returned, the products may still be running

        TAPP_wait_all(et1);
        TAPP_wait_all(et2);
        double t2 = now();   // all products are done
        double ts = t1 - t0;
        double tw = t2 - t1;
        double tt = t2 - t0;

        tsmin = tsmin == 0.0 || tsmin > ts ? ts : tsmin;
        tsmean += ts;

        twmin = twmin == 0.0 || twmin > tw ? tw : twmin;
        twmean += tw;

        ttmin = ttmin == 0.0 || ttmin > tt ? tt : ttmin;
        ttmean += tt;

        TAPP_destroy_status(s1);
        TAPP_destroy_status(s2);
    }

    tsmean /= MEASURED_RUNS;
    twmean /= MEASURED_RUNS;
    ttmean /= MEASURED_RUNS;

    printf("Two executors:\n");
    printf("\tsubmit: min %.3f ms, mean %.3f ms\n", tsmin * 1e3, tsmean * 1e3);
    printf("\twait:   min %.3f ms, mean %.3f ms\n", twmin * 1e3, twmean * 1e3);
    printf("\ttotal:  min %.3f ms, mean %.3f ms\n", ttmin * 1e3, ttmean * 1e3);

    TAPP_destroy_tensor_product(p);

    TAPP_destroy_executor(et1);
    TAPP_destroy_executor(et2);

    TAPP_shutdown();
}

int main(int argc, char *argv[])
{
    size_t size = N * N;
    size_t bytes = size * sizeof(double);
    double* ha = malloc(bytes);
    double* hb = malloc(bytes);
    double* hc = malloc(bytes);
    double* ddr = calloc(size, sizeof(double));
    double* dd_host = calloc(size, sizeof(double));
    if (ha == NULL || hb == NULL || hc == NULL || ddr == NULL || dd_host == NULL)
    {
        fprintf(stderr, "Failed to allocate the tensors\n");
        free(ha);
        free(hb);
        free(hc);
        free(ddr);
        free(dd_host);
        return 1;
    }

    srand(1);
    generate_data(ha, size);
    generate_data(hb, size);
    generate_data(hc, size);

    double* da;
    double* db;
    double* dc;
    double* dd1;
    double* dd2;
    CUDA_CHECK(cudaMalloc((void**)&da, bytes));
    CUDA_CHECK(cudaMalloc((void**)&db, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dc, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dd1, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dd2, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, hc, bytes, cudaMemcpyHostToDevice));

    run_example(da, db, dc, ddr, dd1, dd2, dd_host, size);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(dd1));
    CUDA_CHECK(cudaFree(dd2));

    free(ha);
    free(hb);
    free(hc);
    free(ddr);
    free(dd_host);
    return 0;
}