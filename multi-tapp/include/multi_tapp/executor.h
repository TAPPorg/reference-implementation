#ifndef TAPP_MULTI_TAPP_EXECUTOR_H_
#define TAPP_MULTI_TAPP_EXECUTOR_H_

#include "backend.h"
#include "tapp/executor.h"

TAPP_EXPORT TAPP_error Multi_TAPP_create_executor(Multi_TAPP_backend backend, TAPP_executor *exec, TAPP_attr attr);

TAPP_EXPORT TAPP_error Multi_TAPP_destroy_executor(Multi_TAPP_backend backend, TAPP_executor exec);

TAPP_EXPORT TAPP_error Multi_TAPP_wait_one(Multi_TAPP_backend backend, TAPP_executor exec, TAPP_status status);

TAPP_EXPORT TAPP_error Multi_TAPP_wait_all(Multi_TAPP_backend backend, TAPP_executor exec);

#endif /* TAPP_MULTI_TAPP_EXECUTOR_H_ */