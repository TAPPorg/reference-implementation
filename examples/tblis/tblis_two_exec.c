// CPU masks (TAPP_CPU_MASK) use the Linux affinity API. On other platforms the executors run on any CPU
#if defined(__linux__)
#define _GNU_SOURCE // for cpu_set_t and the CPU_* macros
#include <sched.h>
#endif
#include "tapp_tblis.h"
#include <tapp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/*
 * htop check threads
 * start program
 * htop -p $(pgrep -n -f tapp-tblis-two-exec)
 * (See to that the PROCESSOR column is there)
 * press: H
 * press enter on the program to continue it after the executors have been created
 */

#define WARM_UP_RUNS 5
#define MEASURED_RUNS 5
#define N 1000 // extent of every mode: each product is about 2 * N^3 = 2 GFLOP

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

int main(int argc, char *argv[])
{
    size_t size = N * N;
    double* da = malloc(size * sizeof(double));
    double* db = malloc(size * sizeof(double));
    double* dc = malloc(size * sizeof(double));
    double* ddr = calloc(size, sizeof(double));
    double* dd1 = calloc(size, sizeof(double));
    double* dd2 = calloc(size, sizeof(double));
    if (da == NULL || db == NULL || dc == NULL || ddr == NULL || dd1 == NULL || dd2 == NULL)
    {
        fprintf(stderr, "Failed to allocate the tensors\n");
        free(da);
        free(db);
        free(dc);
        free(ddr);
        free(dd1);
        free(dd2);
        return 1;
    }

    srand(1);
    generate_data(da, size);
    generate_data(db, size);
    generate_data(dc, size);
    
    TAPP_start();

    TAPP_executor et1;
    TAPP_attr a;
    TAPP_create_attr(&a);
#if defined(__linux__)
    cpu_set_t s;
    CPU_ZERO(&s);
    for (int cpu = 0; cpu < 4; cpu++) CPU_SET(cpu, &s);
    TAPP_attr_set(a, TAPP_CPU_MASK, sizeof(cpu_set_t), &s);
#endif
    unsigned trcnt = 4;
    TAPP_attr_set(a, TAPP_THREAD_COUNT, sizeof(unsigned), &trcnt);
    TAPP_create_executor(&et1, a);

    printf("pid %d, press enter\n", getpid()); getchar();

    TAPP_executor et2;
#if defined(__linux__)
    CPU_ZERO(&s);
    for (int cpu = 4; cpu < 8; cpu++) CPU_SET(cpu, &s);
    TAPP_attr_set(a, TAPP_CPU_MASK, sizeof(cpu_set_t), &s);
#endif
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

    free(da);
    free(db);
    free(dc);
    free(ddr);
    free(dd1);
    free(dd2);
    return 0;
}