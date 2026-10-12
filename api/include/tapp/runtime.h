#ifndef TAPP_RUNTIME_H_
#define TAPP_RUNTIME_H_

#include "util.h"
#include "error.h"

/*
 * Start the implementation. Should be called before any other TAPP function: an implementation might need it, and a
 * program can not know which implementation it ends up linked to. Repeated calls do nothing.
 */
TAPP_EXPORT TAPP_error TAPP_start(void);

/*
 * Shut down the implementation. Should be called after the last use of TAPP. Objects that still exist are left
 * untouched. Repeated calls do nothing.
 */
TAPP_EXPORT TAPP_error TAPP_shutdown(void);

#endif /* TAPP_RUNTIME_H_ */
