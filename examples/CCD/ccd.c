#define _GNU_SOURCE // for cpu_set_t and the CPU_* macros
#include <complex.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "multi_tapp.h"
#include "tapp_tblis.h"

enum
{
    NONE = -1,
    // Inputs
    F1,
    F2,
    V1,
    V2,
    V3,
    V4,
    V5,
    T1,
    // Intermediates
    A2,
    D1,
    E2,
    G1,
    G2,
    I1,
    J1,
    M1,
    X1,
    Y1,
    // Results
    Z0,
    Z2,
    // Scalar with the value 1: the second tensor of the terms that only have one
    ONE,
    TENSOR_COUNT
} tensors;

#define MAX_NMODE 4

// Index spaces of each tensor: o = occupied, v = virtual
static const char* const shapes[TENSOR_COUNT] = {
    [F1] = "oo",   [F2] = "vv",
    [V1] = "oovv", [V2] = "vvvv", [V3] = "voov", [V4] = "oooo", [V5] = "vvoo",
    [T1] = "vvoo",
    [A2] = "oo",   [D1] = "vvoo", [E2] = "oooo", [G1] = "vvoo", [G2] = "vvoo",
    [I1] = "vvoo", [J1] = "vv",   [M1] = "vvoo", [X1] = "ovvo", [Y1] = "vvoo",
    [Z0] = "",     [Z2] = "vvoo",
    [ONE] = "",
};

// One term of the equations: d += alpha * a * b
struct term
{
    int d;
    double alpha;
    int a;
    const char* idx_a;
    int b;
    const char* idx_b;
    const char* idx_d;
    int dep;   // position in terms of the last term that writes the intermediate this term reads, or NONE
};

// The terms in the order they are executed: an intermediate is complete before it is used.
// Indices: l m M n = l1 m1 m2 n1 (occupied), d e E f = d1 e1 e2 f1 (virtual)
static const struct term terms[] = {
    {Z0,  0.25, V1, "mnef", T1,  "efmn", ""    , NONE},   //  0
    {D1,  1.0,  F1, "lm",   T1,  "fenl", "femn", NONE},   //  1
    {Z2,  0.5,  V2, "efdE", T1,  "dEmn", "efmn", NONE},   //  2
    {G1,  1.0,  V3, "elmd", T1,  "fdnl", "efmn", NONE},   //  3
    {I1,  1.0,  F2, "ed",   T1,  "fdnm", "efnm", NONE},   //  4
    {J1,  1.0,  V1, "lMdE", T1,  "EflM", "fd"  , NONE},   //  5
    {M1,  1.0,  T1, "edmn", J1,  "fd",   "efmn", 5   },   //  6
    {X1,  1.0,  V1, "lMdE", T1,  "EfMn", "lfdn", NONE},   //  7
    {Y1,  1.0,  T1, "edml", X1,  "lfdn", "efmn", 7   },   //  8
    {A2,  1.0,  V1, "lMdE", T1,  "dEMn", "ln"  , NONE},   //  9
    {E2,  1.0,  V1, "lMdE", T1,  "dEmn", "lMmn", NONE},   // 10
    {E2,  2.0,  V4, "lMmn", ONE, "",     "lMmn", NONE},   // 11
    {G2,  1.0,  T1, "efml", A2,  "ln",   "efmn", 9   },   // 12
    {Z2,  0.25, T1, "eflM", E2,  "lMmn", "efmn", 11  },   // 13
    {Z2,  0.5,  Y1, "efmn", ONE, "",     "efmn", 8   },   // 14
    {Z2, -0.5,  Y1, "femn", ONE, "",     "efmn", 8   },   // 15
    {Z2, -0.5,  Y1, "efnm", ONE, "",     "efmn", 8   },   // 16
    {Z2,  0.5,  Y1, "fenm", ONE, "",     "efmn", 8   },   // 17
    {Z2,  1.0,  G1, "efmn", ONE, "",     "efmn", 3   },   // 18
    {Z2, -1.0,  G1, "femn", ONE, "",     "efmn", 3   },   // 19
    {Z2, -1.0,  G1, "efnm", ONE, "",     "efmn", 3   },   // 20
    {Z2,  1.0,  G1, "fenm", ONE, "",     "efmn", 3   },   // 21
    {Z2,  0.5,  M1, "efmn", ONE, "",     "efmn", 6   },   // 22
    {Z2, -0.5,  M1, "femn", ONE, "",     "efmn", 6   },   // 23
    {Z2,  1.0,  I1, "efnm", ONE, "",     "efmn", 4   },   // 24
    {Z2, -1.0,  I1, "fenm", ONE, "",     "efmn", 4   },   // 25
    {Z2,  0.5,  G2, "efmn", ONE, "",     "efmn", 12  },   // 26
    {Z2, -0.5,  G2, "efnm", ONE, "",     "efmn", 12  },   // 27
    {Z2, -1.0,  D1, "femn", ONE, "",     "efmn", 1   },   // 28
    {Z2,  1.0,  D1, "fenm", ONE, "",     "efmn", 1   },   // 29
    {Z2,  1.0,  V5, "efmn", ONE, "",     "efmn", NONE},   // 30
};

#define TERM_COUNT ((int)(sizeof(terms) / sizeof(terms[0])))

// Index labels as the int64_t array of a tensor product, from one character per index
static void create_indices(const char* labels, int64_t* idx)
{
    for (int i = 0; labels[i] != '\0'; i++) idx[i] = labels[i];
}

// Prints the explanation of a failed call of the backend. Returns nonzero if the call failed.
static int failed(TAPP_backend b, TAPP_error error, const char* what, int term)
{
    if (Multi_TAPP_check_success(b, error)) return 0;
    char message[256];
    Multi_TAPP_explain_error(b, error, sizeof(message), message);
    fprintf(stderr, "%s of term %d failed: %s\n", what, term + 1, message);
    return 1;
}

// Number of elements of a tensor: the product of its extents (1 for a scalar)
static int64_t tensor_size(int nmode, const int64_t* extents)
{
    int64_t size = 1;
    for (int i = 0; i < nmode; i++) size *= extents[i];
    return size;
}

// Column-major strides of a dense tensor
static void create_strides(int nmode, const int64_t* extents, int64_t* strides)
{
    int64_t stride = 1;
    for (int i = 0; i < nmode; i++)
    {
        strides[i] = stride;
        stride *= extents[i];
    }
}

static void create_tensor_info(const int e_occ, const int e_vir, const char* shape, const TAPP_backend b, int64_t* size, TAPP_tensor_info* t) 
{
    int nmode = (int)strlen(shape);
    int64_t ext[MAX_NMODE], str[MAX_NMODE];
    for (int j = 0; j < nmode; j++) ext[j] = shape[j] == 'o' ? e_occ : e_vir;
    create_strides(nmode, ext, str);
    *size = tensor_size(nmode, ext);
    Multi_TAPP_create_tensor_info(b, t, TAPP_C64, nmode, ext, str);
}

// Pseudorandom number in [-1, 1], from the sequence seeded with srand
static double random_double(void)
{
    return 2.0 * rand() / RAND_MAX - 1.0;
}

// Allocates the data of a tensor with size elements: if random is set, every element gets a pseudorandom real and
// imaginary part in [-1, 1], otherwise the elements are zero.
// Returns NULL if the allocation fails. Free with free.
static double complex* generate_data(int64_t size, int random)
{
    double complex* data = calloc((size_t)size, sizeof(double complex));
    if (data == NULL || !random) return data;
    for (int64_t i = 0; i < size; i++)
    {
        double re = random_double();
        double im = random_double();
        data[i] = CMPLX(re, im);
    }
    return data;
}

int main(int argc, char *argv[])
{
    if (argc < 3) return -1;

    int e_occ = (int)strtol(argv[1], NULL, 10);
    int e_vir = (int)strtol(argv[2], NULL, 10);
    int seed = argc >= 4 ? (int)strtol(argv[3], NULL, 10) : rand();
    TAPP_backend backend;
    Multi_TAPP_create_backend(&backend, TAPP_TBLIS_LIBPATH);
    Multi_TAPP_start(backend);

    TAPP_attr attr;
    Multi_TAPP_create_attr(backend, &attr);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    Multi_TAPP_attr_set(backend, attr, TAPP_CPU_MASK, sizeof(cpu_set_t), &cpuset);

    TAPP_executor exec;
    Multi_TAPP_create_executor(backend, &exec, attr);
    
    TAPP_tensor_info tensor_infos[TENSOR_COUNT];

    int64_t size[TENSOR_COUNT];   // number of elements, for allocating the data

    for (int i = 0; i < TENSOR_COUNT; i++)
    {
        create_tensor_info(e_occ, e_vir, shapes[i], backend, &size[i], &tensor_infos[i]);
    }

    // Inputs get random values, intermediates and results start at zero
    double complex* data[TENSOR_COUNT];
    srand(seed);   // fixed seed, so every run gets the same data
    for (int i = 0; i < TENSOR_COUNT; i++)
    {
        data[i] = generate_data(size[i], i < A2);
    }

    data[ONE][0] = 1;

    // One plan per term, created once
    TAPP_tensor_product product_plans[TERM_COUNT];
    for (int i = 0; i < TERM_COUNT; i++)
    {
        const struct term* t = &terms[i];
        int64_t idx_a[MAX_NMODE], idx_b[MAX_NMODE], idx_d[MAX_NMODE];
        create_indices(t->idx_a, idx_a);
        create_indices(t->idx_b, idx_b);
        create_indices(t->idx_d, idx_d);
        TAPP_error error = Multi_TAPP_create_tensor_product(backend, &product_plans[i], exec,
                                                            TAPP_IDENTITY, tensor_infos[t->a], idx_a,
                                                            TAPP_IDENTITY, tensor_infos[t->b], idx_b,
                                                            TAPP_IDENTITY, tensor_infos[t->d], idx_d,
                                                            TAPP_IDENTITY, tensor_infos[t->d], idx_d,
                                                            TAPP_DEFAULT_PREC);
        if (failed(backend, error, "creating the plan", i)) return 1;
    }

    TAPP_status statuses[TERM_COUNT];
    TAPP_status completed = 0;
    // d = alpha * a * b + beta * c, with c = d and beta = 1 to add to the output.
    // The executor runs the terms one after the other in the order they are submitted.
    for (int i = 0; i < TERM_COUNT; i++)
    {
        const struct term* t = &terms[i];
        if (t->dep != NONE)
        {
            if (completed < statuses[t->dep])
            {
                Multi_TAPP_wait_one(backend, exec, statuses[t->dep]);
                completed = statuses[t->dep];
            }
        }
        double complex alpha = t->alpha;
        double complex beta = 1;
        TAPP_error error = Multi_TAPP_execute_product(backend, product_plans[i], exec, &statuses[i], &alpha,
                                                      data[t->a], data[t->b], &beta, data[t->d], data[t->d]);
        if (failed(backend, error, "executing", i)) return 1;
    }

    // Executing only submits the terms: wait until they have run before the results are read
    Multi_TAPP_wait_all(backend, exec);

    printf("Z0 = %.12g + %.12gi\n", creal(data[Z0][0]), cimag(data[Z0][0]));

    for (int i = 0; i < TERM_COUNT; i++) Multi_TAPP_destroy_tensor_product(backend, product_plans[i]);

    for (int i = 0; i < TENSOR_COUNT; i++) free(data[i]);

    for (int i = 0; i < TENSOR_COUNT; i++) Multi_TAPP_destroy_tensor_info(backend, tensor_infos[i]);

    Multi_TAPP_destroy_executor(backend, exec);
    Multi_TAPP_destroy_attr(backend, attr);

    Multi_TAPP_shutdown(backend);
    Multi_TAPP_destroy_backend(backend);

    return 0;
}

