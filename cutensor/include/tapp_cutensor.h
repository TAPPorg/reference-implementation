#ifndef TAPP_CUTENSOR_H_
#define TAPP_CUTENSOR_H_

/*
 * Public definitions of the cuTENSOR implementation of TAPP, beyond the common interface in tapp.h. They are specific
 * to this implementation: other implementations do not know them.
 */

#include <tapp.h>

/*
 * Tensor data passed to TAPP_execute_product and TAPP_execute_permute must be device memory aligned to 128 bytes, as
 * cudaMalloc returns it. A pointer into the middle of an allocation, such as to a subtensor, usually is not: the execute
 * functions refuse it with an error.
 */

/*
 * Plans and statuses belong to the device of the executor they were created with. Executing a plan or waiting for a
 * status with an executor on another device is refused with an error.
 */

/*
 * Attribute keys for TAPP_attr_set. An executor reads its attributes when it is created.
 */
enum
{
    /*
     * One cudaStream_t: the stream the executor runs its operations on. The executor does not take ownership of it,
     * the caller destroys it after the executor. Without it the executor creates and owns a stream of its own.
     */
    TAPP_CUTENSOR_STREAM = 0,

    /*
     * One int: the CUDA device the executor runs on. Without it the executor uses the device of TAPP_CUTENSOR_STREAM
     * (CUDA 12.8 or later), or device 0.
     */
    TAPP_CUTENSOR_DEVICE = 1
};

#endif  /* TAPP_CUTENSOR_H_ */
