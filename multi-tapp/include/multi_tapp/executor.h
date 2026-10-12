#ifndef TAPP_MULTI_TAPP_EXECUTOR_H_
#define TAPP_MULTI_TAPP_EXECUTOR_H_

#include "backend.h"
#include "tapp/executor.h"

TAPP_EXPORT TAPP_error Multi_TAPP_create_executor(TAPP_backend backend, TAPP_executor *exec, TAPP_attr attr);

TAPP_EXPORT TAPP_error Multi_TAPP_destroy_executor(TAPP_backend backend, TAPP_executor exec);

TAPP_EXPORT TAPP_error Multi_TAPP_wait_one(TAPP_backend backend, TAPP_executor exec, TAPP_status status);

TAPP_EXPORT TAPP_error Multi_TAPP_wait_all(TAPP_backend backend, TAPP_executor exec);

#endif /* TAPP_MULTI_TAPP_EXECUTOR_H_ */