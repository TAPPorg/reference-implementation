#ifndef TAPP_TBLIS_H_
#define TAPP_TBLIS_H_

/*
 * Public definitions of the TBLIS implementation of TAPP, beyond the common interface in tapp.h. They are specific to
 * this implementation: other implementations do not know them.
 */

#include <tapp.h>

/*
 * Attribute keys for TAPP_attr_set. An executor reads its attributes when it is created.
 */
enum
{
    /*
     * One int: the number of threads the executor uses for an operation. Without it the executor uses one thread per
     * CPU of TAPP_CPU_MASK, or the default of TBLIS if that is not set either.
     */
    TAPP_THREAD_COUNT = 0,

    /*
     * One cpu_set_t: the CPUs the threads of the executor run on. It takes precedence over the thread binding of
     * OpenMP (OMP_PROC_BIND, OMP_PLACES).
     */
    TAPP_CPU_MASK = 1
};

#endif  /* TAPP_TBLIS_H_ */
