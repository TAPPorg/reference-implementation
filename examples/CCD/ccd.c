#define _GNU_SOURCE // for cpu_set_t and the CPU_* macros
#include <complex.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "multi_tapp.h"
#include "tapp_tblis.h"
#include "run_file.h"

enum
{
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

// Names of the tensors, for messages
static const char* const names[TENSOR_COUNT] = {
    [F1] = "F1", [F2] = "F2",
    [V1] = "V1", [V2] = "V2", [V3] = "V3", [V4] = "V4", [V5] = "V5",
    [T1] = "T1",
    [A2] = "A2", [D1] = "D1", [E2] = "E2", [G1] = "G1", [G2] = "G2",
    [I1] = "I1", [J1] = "J1", [M1] = "M1", [X1] = "X1", [Y1] = "Y1",
    [Z0] = "Z0", [Z2] = "Z2",
    [ONE] = "ONE",
};

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
};

// The terms in the order of the equations: an intermediate is complete before it is used. The run file calls them c1
// to c31 and may run them in another order, as long as that computes the same (see check_order).
// Indices: l m M n = l1 m1 m2 n1 (occupied), d e E f = d1 e1 e2 f1 (virtual)
static const struct term terms[] = {
    {Z0,  0.25, V1, "mnef", T1,  "efmn", ""    },   // c1
    {D1,  1.0,  F1, "lm",   T1,  "fenl", "femn"},   // c2
    {Z2,  0.5,  V2, "efdE", T1,  "dEmn", "efmn"},   // c3
    {G1,  1.0,  V3, "elmd", T1,  "fdnl", "efmn"},   // c4
    {I1,  1.0,  F2, "ed",   T1,  "fdnm", "efnm"},   // c5
    {J1,  1.0,  V1, "lMdE", T1,  "EflM", "fd"  },   // c6
    {M1,  1.0,  T1, "edmn", J1,  "fd",   "efmn"},   // c7
    {X1,  1.0,  V1, "lMdE", T1,  "EfMn", "lfdn"},   // c8
    {Y1,  1.0,  T1, "edml", X1,  "lfdn", "efmn"},   // c9
    {A2,  1.0,  V1, "lMdE", T1,  "dEMn", "ln"  },   // c10
    {E2,  1.0,  V1, "lMdE", T1,  "dEmn", "lMmn"},   // c11
    {E2,  2.0,  V4, "lMmn", ONE, "",     "lMmn"},   // c12
    {G2,  1.0,  T1, "efml", A2,  "ln",   "efmn"},   // c13
    {Z2,  0.25, T1, "eflM", E2,  "lMmn", "efmn"},   // c14
    {Z2,  0.5,  Y1, "efmn", ONE, "",     "efmn"},   // c15
    {Z2, -0.5,  Y1, "femn", ONE, "",     "efmn"},   // c16
    {Z2, -0.5,  Y1, "efnm", ONE, "",     "efmn"},   // c17
    {Z2,  0.5,  Y1, "fenm", ONE, "",     "efmn"},   // c18
    {Z2,  1.0,  G1, "efmn", ONE, "",     "efmn"},   // c19
    {Z2, -1.0,  G1, "femn", ONE, "",     "efmn"},   // c20
    {Z2, -1.0,  G1, "efnm", ONE, "",     "efmn"},   // c21
    {Z2,  1.0,  G1, "fenm", ONE, "",     "efmn"},   // c22
    {Z2,  0.5,  M1, "efmn", ONE, "",     "efmn"},   // c23
    {Z2, -0.5,  M1, "femn", ONE, "",     "efmn"},   // c24
    {Z2,  1.0,  I1, "efnm", ONE, "",     "efmn"},   // c25
    {Z2, -1.0,  I1, "fenm", ONE, "",     "efmn"},   // c26
    {Z2,  0.5,  G2, "efmn", ONE, "",     "efmn"},   // c27
    {Z2, -0.5,  G2, "efnm", ONE, "",     "efmn"},   // c28
    {Z2, -1.0,  D1, "femn", ONE, "",     "efmn"},   // c29
    {Z2,  1.0,  D1, "fenm", ONE, "",     "efmn"},   // c30
    {Z2,  1.0,  V5, "efmn", ONE, "",     "efmn"},   // c31
};

#define TERM_COUNT ((int)(sizeof(terms) / sizeof(terms[0])))

// Checks that the steps of the run file compute the same as the terms in their own order: two terms keep their order
// if one of them reads the tensor the other writes. Terms that only write the same tensor may swap, since they add to
// it. Returns 0 if the order is fine, otherwise the first two terms in the wrong order are printed.
static int check_order(const struct run_file* run)
{
    for (int i = 0; i < run->step_count; i++)
    {
        for (int j = i + 1; j < run->step_count; j++)
        {
            // first is run before second, but comes after it in terms
            int first = run->steps[i].term;
            int second = run->steps[j].term;
            if (first < second) continue;
            const struct term* f = &terms[first];
            const struct term* s = &terms[second];
            if (f->a == s->d || f->b == s->d)
            {
                fprintf(stderr, "term c%d is run before term c%d, but it reads %s, which c%d writes\n",
                        first + 1, second + 1, names[s->d], second + 1);
                return 1;
            }
            if (s->a == f->d || s->b == f->d)
            {
                fprintf(stderr, "term c%d is run before term c%d, but it writes %s, which c%d reads without that\n",
                        first + 1, second + 1, names[f->d], second + 1);
                return 1;
            }
        }
    }
    return 0;
}

static int max_int(int a, int b)
{
    return a > b ? a : b;
}

// Index labels as the int64_t array of a tensor product, from one character per index
static void create_indices(const char* labels, int64_t* idx)
{
    for (int i = 0; labels[i] != '\0'; i++) idx[i] = labels[i];
}

// Prints the explanation of a failed call of the backend. Returns nonzero if the call failed.
static int failed(Multi_TAPP_backend b, TAPP_error error, const char* what, int term)
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

// Extents of a tensor from its index spaces. Returns the number of modes.
static int shape_extents(int e_occ, int e_vir, const char* shape, int64_t* extents)
{
    int nmode = (int)strlen(shape);
    for (int i = 0; i < nmode; i++) extents[i] = shape[i] == 'o' ? e_occ : e_vir;
    return nmode;
}

static void create_tensor_info(int e_occ, int e_vir, const char* shape, Multi_TAPP_backend backend, TAPP_tensor_info* info)
{
    int64_t ext[MAX_NMODE], str[MAX_NMODE];
    int nmode = shape_extents(e_occ, e_vir, shape, ext);
    create_strides(nmode, ext, str);
    Multi_TAPP_create_tensor_info(backend, info, TAPP_C64, nmode, ext, str);
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
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <run file> <occupied> <virtual> [<seed>]\n", argv[0]);
        return 1;
    }

    const char* run_path = argv[1];
    int e_occ = (int)strtol(argv[2], NULL, 10);
    int e_vir = (int)strtol(argv[3], NULL, 10);
    int seed = argc >= 5 ? (int)strtol(argv[4], NULL, 10) : rand();

    struct run_file run;
    if (read_run_file(run_path, TERM_COUNT, &run) != 0) return 1;
    if (check_order(&run) != 0)
    {
        free_run_file(&run);
        return 1;
    }

    Multi_TAPP_backend* backends = malloc(run.backend_count * sizeof(Multi_TAPP_backend));

    for (int i = 0; i < run.backend_count; i++)
    {
        Multi_TAPP_create_backend(&backends[i], run.backends[i].path);
        Multi_TAPP_start(backends[i]);
    }

    TAPP_executor* executors = malloc(run.executor_count * sizeof(TAPP_executor));

    for (int i = 0; i < run.executor_count; i++)
    {
        Multi_TAPP_backend backend = backends[run.executors[i].backend];
        TAPP_attr attr = 0;
        if (run.executors[i].has_cpus)
        {
            Multi_TAPP_create_attr(backend, &attr);
            Multi_TAPP_attr_set(backend, attr, TAPP_CPU_MASK, sizeof(cpu_set_t), &run.executors[i].cpus);
        }
        Multi_TAPP_create_executor(backend, &executors[i], attr);
        if (run.executors[i].has_cpus)
        {
            Multi_TAPP_destroy_attr(backend, attr);
        }
    }
    
    int64_t sizes[TENSOR_COUNT];   // number of elements, for allocating the data
    for (int i = 0; i < TENSOR_COUNT; i++)
    {
        int64_t ext[MAX_NMODE];
        int nmode = shape_extents(e_occ, e_vir, shapes[i], ext);
        sizes[i] = tensor_size(nmode, ext);
    }

    // A tensor info belongs to one backend: create the ones each backend needs for its terms.
    // 0 marks a tensor info that is not created.
    TAPP_tensor_info tensor_infos[run.backend_count][TENSOR_COUNT];
    memset(tensor_infos, 0, sizeof(tensor_infos));
    for (int i = 0; i < run.step_count; i++)
    {
        const struct term* t = &terms[run.steps[i].term];
        int backend_index = run.executors[run.steps[i].executor].backend;
        int used[3] = {t->a, t->b, t->d};
        for (int j = 0; j < 3; j++)
        {
            TAPP_tensor_info* info = &tensor_infos[backend_index][used[j]];
            if (*info != 0) continue;
            create_tensor_info(e_occ, e_vir, shapes[used[j]], backends[backend_index], info);
        }
    }

    // Inputs get random values, intermediates and results start at zero
    double complex* data[TENSOR_COUNT];
    srand(seed);   // fixed seed, so every run gets the same data
    for (int i = 0; i < TENSOR_COUNT; i++)
    {
        data[i] = generate_data(sizes[i], i < A2);
    }

    data[ONE][0] = 1;

    // One plan per term, created once
    TAPP_tensor_product product_plans[run.step_count];
    for (int i = 0; i < run.step_count; i++)
    {
        const struct term* t = &terms[run.steps[i].term];
        int backend_index = run.executors[run.steps[i].executor].backend;
        int executor_index = run.steps[i].executor;
        int64_t idx_a[MAX_NMODE], idx_b[MAX_NMODE], idx_d[MAX_NMODE];
        create_indices(t->idx_a, idx_a);
        create_indices(t->idx_b, idx_b);
        create_indices(t->idx_d, idx_d);
        TAPP_error error = Multi_TAPP_create_tensor_product(backends[backend_index], &product_plans[i], executors[executor_index],
                                                            TAPP_IDENTITY, tensor_infos[backend_index][t->a], idx_a,
                                                            TAPP_IDENTITY, tensor_infos[backend_index][t->b], idx_b,
                                                            TAPP_IDENTITY, tensor_infos[backend_index][t->d], idx_d,
                                                            TAPP_IDENTITY, tensor_infos[backend_index][t->d], idx_d,
                                                            TAPP_DEFAULT_PREC);
        if (failed(backends[backend_index], error, "creating the plan", run.steps[i].term)) return 1;
    }

    // An executor runs its steps one after the other in the order they are submitted, but the executors run at the
    // same time. A step therefore waits for the steps on the other executors that it must not overlap with: the ones
    // that write a tensor it uses, and the ones that read the tensor it writes.
    // Steps are numbered from 1 here, so that 0 is no step.
    TAPP_status statuses[run.step_count];
    int last_write[TENSOR_COUNT][run.executor_count];   // last step on the executor that writes the tensor
    int last_read[TENSOR_COUNT][run.executor_count];    // last step on the executor that reads the tensor
    int completed[run.executor_count];                  // last step on the executor that is known to be done
    memset(last_write, 0, sizeof(last_write));
    memset(last_read, 0, sizeof(last_read));
    memset(completed, 0, sizeof(completed));

    // d = alpha * a * b + beta * c, with c = d and beta = 1 to add to the output
    for (int i = 0; i < run.step_count; i++)
    {
        int executor_index = run.steps[i].executor;
        int backend_index = run.executors[executor_index].backend;
        int term_index = run.steps[i].term;
        const struct term* t = &terms[term_index];

        for (int e = 0; e < run.executor_count; e++)
        {
            if (e == executor_index) continue;
            int step = max_int(max_int(last_write[t->a][e], last_write[t->b][e]),
                               max_int(last_write[t->d][e], last_read[t->d][e]));
            if (step <= completed[e]) continue;
            Multi_TAPP_wait_one(backends[run.executors[e].backend], executors[e], statuses[step - 1]);
            completed[e] = step;
        }

        double complex alpha = t->alpha;
        double complex beta = 1;
        TAPP_error error = Multi_TAPP_execute_product(backends[backend_index], product_plans[i], executors[executor_index], &statuses[i], &alpha,
                                                      data[t->a], data[t->b], &beta, data[t->d], data[t->d]);
        if (failed(backends[backend_index], error, "executing", term_index)) return 1;

        last_read[t->a][executor_index] = i + 1;
        last_read[t->b][executor_index] = i + 1;
        last_write[t->d][executor_index] = i + 1;
    }

    for (int i = 0; i < run.executor_count; i++)
    {
        Multi_TAPP_wait_all(backends[run.executors[i].backend], executors[i]);
    }

    printf("Z0 = %.12g + %.12gi\n", creal(data[Z0][0]), cimag(data[Z0][0]));

    // A plan belongs to the backend of the executor its step runs on
    for (int i = 0; i < run.step_count; i++)
    {
        int backend_index = run.executors[run.steps[i].executor].backend;
        Multi_TAPP_destroy_tensor_product(backends[backend_index], product_plans[i]);
    }

    for (int i = 0; i < TENSOR_COUNT; i++) free(data[i]);

    for (int i = 0; i < run.backend_count; i++)
    {
        for (int j = 0; j < TENSOR_COUNT; j++)
        {
            if (tensor_infos[i][j] != 0) Multi_TAPP_destroy_tensor_info(backends[i], tensor_infos[i][j]);
        }
    }

    for (int i = 0; i < run.executor_count; i++)
    {
        Multi_TAPP_destroy_executor(backends[run.executors[i].backend], executors[i]);
    }
    free(executors);

    for (int i = 0; i < run.backend_count; i++)
    {
        Multi_TAPP_shutdown(backends[i]);
        Multi_TAPP_destroy_backend(backends[i]);
    }
    free(backends);

    free_run_file(&run);

    return 0;
}

