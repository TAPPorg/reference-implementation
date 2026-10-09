#ifndef TAPP_TBLIS_EXECUTOR_H_
#define TAPP_TBLIS_EXECUTOR_H_

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <tapp.h>
#include <thread>

#include "tapp/error.h"
#include "tblis_utils.h"

struct executor
{
    std::thread worker;
    std::queue<std::function<TAPP_error()>> tasks;

    std::mutex m;
    std::condition_variable cv;
    std::condition_variable done_cv;

    uint64_t completed = 0;
    uint64_t submitted = 0;

    bool stopping = false;
};

uint64_t submit_task(executor* exec, std::function<TAPP_error()> task);

#endif  /* TAPP_TBLIS_EXECUTOR_H_ */
