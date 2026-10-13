#ifndef TAPP_LIFECYCLE_H_
#define TAPP_LIFECYCLE_H_

#include "util.h"
#include "error.h"

/*
 * Start the implementation.
 */
TAPP_EXPORT TAPP_error TAPP_start(void);

/*
 * Shut down the implementation.
 */
TAPP_EXPORT TAPP_error TAPP_shutdown(void);

#endif /* TAPP_LIFECYCLE_H_ */
