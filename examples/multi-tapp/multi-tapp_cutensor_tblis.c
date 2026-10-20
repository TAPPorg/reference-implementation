// CPU masks (TAPP_CPU_MASK) use the Linux affinity API. On other platforms the executor runs on any CPU
#if defined(__linux__)
#define _GNU_SOURCE // for cpu_set_t and the CPU_* macros
#include <sched.h>
#endif
#include "multi_tapp.h"
#include "tapp_tblis.h"
#include "tapp_cutensor.h"
#include <cuda_runtime.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * One product on two backends at once through multi-tapp: the same product runs on TBLIS (on the CPU, tensors in
 * host memory) and on cuTENSOR (on the GPU, tensors in device memory). Both are submitted before either is waited for,
 * so they run side by side and the total time approaches that of the slower one instead of the sum of both.
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

// Minimum and mean of the measured times in seconds
struct timing
{
    double min;
    double mean;
};

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void add_time(struct timing* t, double time)
{
    t->min = t->min == 0.0 || t->min > time ? time : t->min;
    t->mean += time / MEASURED_RUNS;
}

static void print_timing(const char* name, struct timing t)
{
    printf("%-22s min %8.3f ms, mean %8.3f ms\n", name, t.min * 1e3, t.mean * 1e3);
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

// Computes the result of the product with the reference implementation into ddr. All tensors are in host memory.
void run_reference(const double* da, const double* db, const double* dc, double* ddr)
{

    Multi_TAPP_backend refb;
    
    Multi_TAPP_create_backend(&refb, TAPP_REFERENCE_LIBPATH);

    Multi_TAPP_start(refb);

    TAPP_executor refe;
    Multi_TAPP_create_executor(refb, &refe, 0);

    TAPP_tensor_info refta;
    Multi_TAPP_create_tensor_info(refb, &refta, TAPP_DOUBLE, 2, ext, strd);
    TAPP_tensor_info reftb;
    Multi_TAPP_create_tensor_info(refb, &reftb, TAPP_DOUBLE, 2, ext, strd);
    TAPP_tensor_info reftd;
    Multi_TAPP_create_tensor_info(refb, &reftd, TAPP_DOUBLE, 2, ext, strd);

    TAPP_tensor_product refp;
    Multi_TAPP_create_tensor_product(refb, &refp, refe, TAPP_IDENTITY, refta, idxa, TAPP_IDENTITY, reftb, idxb, TAPP_IDENTITY, reftd, idxd, TAPP_IDENTITY, reftd, idxd, TAPP_DEFAULT_PREC);

    Multi_TAPP_destroy_tensor_info(refb, refta);
    Multi_TAPP_destroy_tensor_info(refb, reftb);
    Multi_TAPP_destroy_tensor_info(refb, reftd);

    Multi_TAPP_execute_product(refb, refp, refe, NULL, &alpha, da, db, &beta, dc, ddr); // the reference implementation is synchronous: the product is done when this returns
    
    Multi_TAPP_destroy_tensor_product(refb, refp);
    Multi_TAPP_destroy_executor(refb, refe);
    Multi_TAPP_shutdown(refb);
    Multi_TAPP_destroy_backend(refb);
}

// Times the product on TBLIS alone, on cuTENSOR alone and on both side by side, and checks the results against ddr.
// ha, hb, hc, hd and ddr are in host memory, da, db, dc and dd in device memory. dd_host receives the copy of dd.
void run_example(const double* ha, const double* hb, const double* hc, double* hd,
                 const double* da, const double* db, const double* dc, double* dd,
                 const double* ddr, double* dd_host, size_t size)
{
    // Create TBLIS structures
    
    Multi_TAPP_backend tbb;

    Multi_TAPP_create_backend(&tbb, TAPP_TBLIS_LIBPATH);
    Multi_TAPP_start(tbb);

    TAPP_executor tbe;
    TAPP_attr tbat;
    Multi_TAPP_create_attr(tbb, &tbat);

#if defined(__linux__)
    // Choose the CPUs the threads run on: the first four of those this process may use
    cpu_set_t usable;
    sched_getaffinity(0, sizeof(cpu_set_t), &usable);
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (int cpu = 0; cpu < CPU_SETSIZE && CPU_COUNT(&cpu_set) < 4; cpu++)
    {
        if (CPU_ISSET(cpu, &usable)) CPU_SET(cpu, &cpu_set);
    }
    Multi_TAPP_attr_set(tbb, tbat, TAPP_CPU_MASK, sizeof(cpu_set_t), &cpu_set);
#endif

    Multi_TAPP_create_executor(tbb, &tbe, tbat);
    Multi_TAPP_destroy_attr(tbb, tbat);

    TAPP_tensor_info tbta;
    Multi_TAPP_create_tensor_info(tbb, &tbta, TAPP_DOUBLE, 2, ext, strd);
    TAPP_tensor_info tbtb;
    Multi_TAPP_create_tensor_info(tbb, &tbtb, TAPP_DOUBLE, 2, ext, strd);
    TAPP_tensor_info tbtd;
    Multi_TAPP_create_tensor_info(tbb, &tbtd, TAPP_DOUBLE, 2, ext, strd);

    TAPP_tensor_product tbp;
    Multi_TAPP_create_tensor_product(tbb, &tbp, tbe, TAPP_IDENTITY, tbta, idxa, TAPP_IDENTITY, tbtb, idxb, TAPP_IDENTITY, tbtd, idxd, TAPP_IDENTITY, tbtd, idxd, TAPP_DEFAULT_PREC);

    Multi_TAPP_destroy_tensor_info(tbb, tbta);
    Multi_TAPP_destroy_tensor_info(tbb, tbtb);
    Multi_TAPP_destroy_tensor_info(tbb, tbtd);

    // Create cuTensor structures
    
    Multi_TAPP_backend cub;

    Multi_TAPP_create_backend(&cub, TAPP_CUTENSOR_LIBPATH);
    Multi_TAPP_start(cub);

    TAPP_executor cue;
    TAPP_attr cuat;

    Multi_TAPP_create_attr(cub, &cuat);
    int cudev = 0;
    Multi_TAPP_attr_set(cub, cuat, TAPP_CUTENSOR_DEVICE, sizeof(int), &cudev);
    Multi_TAPP_create_executor(cub, &cue, cuat);
    Multi_TAPP_destroy_attr(cub, cuat);

    TAPP_tensor_info cuta;
    Multi_TAPP_create_tensor_info(cub, &cuta, TAPP_DOUBLE, 2, ext, strd);
    TAPP_tensor_info cutb;
    Multi_TAPP_create_tensor_info(cub, &cutb, TAPP_DOUBLE, 2, ext, strd);
    TAPP_tensor_info cutd;
    Multi_TAPP_create_tensor_info(cub, &cutd, TAPP_DOUBLE, 2, ext, strd);

    TAPP_tensor_product cup;
    Multi_TAPP_create_tensor_product(cub, &cup, cue, TAPP_IDENTITY, cuta, idxa, TAPP_IDENTITY, cutb, idxb, TAPP_IDENTITY, cutd, idxd, TAPP_IDENTITY, cutd, idxd, TAPP_DEFAULT_PREC);

    Multi_TAPP_destroy_tensor_info(cub, cuta);
    Multi_TAPP_destroy_tensor_info(cub, cutb);
    Multi_TAPP_destroy_tensor_info(cub, cutd);

    // Warmup
    for (size_t i = 0; i < WARM_UP_RUNS; i++)
    {
        Multi_TAPP_execute_product(tbb, tbp, tbe, NULL, &alpha, ha, hb, &beta, hc, hd);
        Multi_TAPP_execute_product(cub, cup, cue, NULL, &alpha, da, db, &beta, dc, dd);
        Multi_TAPP_wait_all(tbb, tbe);
        Multi_TAPP_wait_all(cub, cue);
    }

    struct timing t_tblis = {0, 0};
    struct timing t_cutensor = {0, 0};
    struct timing t_both = {0, 0};
    struct timing t_submit = {0, 0};

    for (size_t i = 0; i < MEASURED_RUNS; i++)
    {
        double t0 = now();
        Multi_TAPP_execute_product(tbb, tbp, tbe, NULL, &alpha, ha, hb, &beta, hc, hd);
        Multi_TAPP_wait_all(tbb, tbe);
        double t1 = now();
        Multi_TAPP_execute_product(cub, cup, cue, NULL, &alpha, da, db, &beta, dc, dd);
        Multi_TAPP_wait_all(cub, cue);
        double t2 = now();
        add_time(&t_tblis, t1 - t0);
        add_time(&t_cutensor, t2 - t1);
    }

    for (size_t i = 0; i < MEASURED_RUNS; i++)
    {
        double t0 = now();
        Multi_TAPP_execute_product(tbb, tbp, tbe, NULL, &alpha, ha, hb, &beta, hc, hd);
        Multi_TAPP_execute_product(cub, cup, cue, NULL, &alpha, da, db, &beta, dc, dd);
        double t1 = now();   // the calls have returned, the products may still be running

        Multi_TAPP_wait_all(tbb, tbe);
        Multi_TAPP_wait_all(cub, cue);
        double t2 = now();   // both products are done
        add_time(&t_submit, t1 - t0);
        add_time(&t_both, t2 - t0);

        if (!same_data(ddr, hd, size, 1e-10)) printf("TBLIS result does not match the reference\n");
        CUDA_CHECK(cudaMemcpy(dd_host, dd, size * sizeof(double), cudaMemcpyDeviceToHost));
        if (!same_data(ddr, dd_host, size, 1e-10)) printf("cuTENSOR result does not match the reference\n");
    }

    print_timing("TBLIS alone:", t_tblis);
    print_timing("cuTENSOR alone:", t_cutensor);
    printf("\n");
    print_timing("Side by side, submit:", t_submit);
    print_timing("Side by side, total:", t_both);
    printf("\nSide by side takes %.0f%% of running them one after the other (mean)\n",
           100.0 * t_both.mean / (t_tblis.mean + t_cutensor.mean));

    // Destroy TBLIS structure

    Multi_TAPP_destroy_tensor_product(tbb, tbp);
    Multi_TAPP_destroy_executor(tbb, tbe);
    Multi_TAPP_shutdown(tbb);
    Multi_TAPP_destroy_backend(tbb);

    // Destroy cuTensor structure

    Multi_TAPP_destroy_tensor_product(cub, cup);
    Multi_TAPP_destroy_executor(cub, cue);
    Multi_TAPP_shutdown(cub);
    Multi_TAPP_destroy_backend(cub);
}

int main(int argc, char *argv[])
{
    size_t size = N * N;
    size_t bytes = size * sizeof(double);
    double* ha = malloc(bytes);
    double* hb = malloc(bytes);
    double* hc = malloc(bytes);
    double* hd = calloc(size, sizeof(double));
    double* ddr = calloc(size, sizeof(double));
    double* dd_host = calloc(size, sizeof(double));
    if (ha == NULL || hb == NULL || hc == NULL || hd == NULL || ddr == NULL || dd_host == NULL)
    {
        fprintf(stderr, "Failed to allocate the tensors\n");
        free(ha);
        free(hb);
        free(hc);
        free(hd);
        free(ddr);
        free(dd_host);
        return 1;
    }

    srand(1);
    generate_data(ha, size);
    generate_data(hb, size);
    generate_data(hc, size);

    run_reference(ha, hb, hc, ddr);

    double* da;
    double* db;
    double* dc;
    double* dd;
    CUDA_CHECK(cudaMalloc((void**)&da, bytes));
    CUDA_CHECK(cudaMalloc((void**)&db, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dc, bytes));
    CUDA_CHECK(cudaMalloc((void**)&dd, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc, hc, bytes, cudaMemcpyHostToDevice));

    run_example(ha, hb, hc, hd, da, db, dc, dd, ddr, dd_host, size);

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(dd));

    free(ha);
    free(hb);
    free(hc);
    free(hd);
    free(ddr);
    free(dd_host);
    return 0;
}
