#ifndef TAPP_CCD_RUN_FILE_H_
#define TAPP_CCD_RUN_FILE_H_

/*
 * Reads a run file: which backends and executors exist, and which executor runs which term in what order. See ccd.run
 * for the format. Nothing of TAPP is used here: the result only describes what to create.
 *
 * cpu_set_t needs _GNU_SOURCE to be defined before the first system header is included.
 */

#include <limits.h>
#include <sched.h>

#define RUN_NAME_MAX 32

struct run_backend
{
    char name[RUN_NAME_MAX];
    char path[PATH_MAX];   // shared library of the TAPP implementation
};

struct run_executor
{
    char name[RUN_NAME_MAX];
    int backend;      // position in backends
    int has_cpus;     // nonzero if cpus is set
    cpu_set_t cpus;
    int threads;      // 0 if not set
};

struct run_step
{
    int term;         // position of the term, starting at 0
    int executor;     // position in executors
};

struct run_file
{
    int backend_count;
    int executor_count;
    int step_count;   // term_count after a successful read
    struct run_backend* backends;
    struct run_executor* executors;
    struct run_step* steps;   // in the order the terms are submitted
};

// Reads the run file at path for term_count terms, which the file calls c1 to c<term_count>.
// Returns 0 on success. Otherwise the problem is printed to stderr and run is left empty.
// Free with free_run_file.
int read_run_file(const char* path, int term_count, struct run_file* run);

void free_run_file(struct run_file* run);

#endif /* TAPP_CCD_RUN_FILE_H_ */
