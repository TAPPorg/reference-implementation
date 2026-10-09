/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "../include/executor.h"
#include "tapp/executor.h"
#include <cmath>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

void worker_loop(executor* exec) {
    while (true) {
        std::function<TAPP_error()> task;

        {
            std::unique_lock<std::mutex> lk(exec->m);
            exec->cv.wait(lk, [exec]{return !exec->tasks.empty() || exec->stopping;});
            if (exec->stopping && exec->tasks.empty()) break;
            task = std::move(exec->tasks.front());
            exec->tasks.pop();
        }
        // TODO: Handle error, also try catch errors
        task();

        {
            std::lock_guard<std::mutex> lk(exec->m);
            exec->completed++;
        }
        exec->done_cv.notify_all();
    }
}

uint64_t submit_task(executor* exec, std::function<TAPP_error()> task) {
    uint64_t id;
    {
        std::lock_guard<std::mutex> lk(exec->m);
        exec->tasks.push(std::move(task));
        id = ++exec->submitted;
    }
    exec->cv.notify_one();
    return id;
}

TAPP_error TAPP_create_executor(TAPP_executor* exec) {
    executor* backend_exec = new executor();
    backend_exec->worker = std::thread(worker_loop, backend_exec);
    *exec = (TAPP_executor) backend_exec;
    return 0;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec) {
    executor* backend_exec = (executor*)exec;

    {
        std::lock_guard<std::mutex> lk(backend_exec->m);
        backend_exec->stopping = true;
    }
    backend_exec->cv.notify_one();

    backend_exec->worker.join();
    delete backend_exec;
    return 0;
}

TAPP_error TAPP_wait_executor(TAPP_executor exec) {
    executor* backend_exec = (executor*)exec;
    std::unique_lock<std::mutex> lk(backend_exec->m);
    backend_exec->done_cv.wait(lk, [backend_exec]{ return backend_exec->completed == backend_exec->submitted;});
    return 0;
}

TAPP_error TAPP_wait_status(TAPP_executor exec, TAPP_status status) {
    executor* backend_exec = (executor*)exec;
    std::unique_lock<std::mutex> lk(backend_exec->m);
    backend_exec->done_cv.wait(lk, [backend_exec, status]{return backend_exec->completed >= status;});
    return 0;
}