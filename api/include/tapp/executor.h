#ifndef TAPP_EXECUTOR_H_
#define TAPP_EXECUTOR_H_

#include <stdint.h>

#include "attributes.h"
#include "util.h"
#include "error.h"
#include "status.h"

typedef intptr_t TAPP_executor;

TAPP_EXPORT TAPP_error TAPP_create_executor(TAPP_executor *exec, TAPP_attr attr);

TAPP_EXPORT TAPP_error TAPP_destroy_executor(TAPP_executor exec);

TAPP_EXPORT TAPP_error TAPP_wait_one(TAPP_executor exec, TAPP_status status);

TAPP_EXPORT TAPP_error TAPP_wait_all(TAPP_executor exec);

#endif /* TAPP_EXECUTOR_H_ */
