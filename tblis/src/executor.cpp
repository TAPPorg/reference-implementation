/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "tapp/executor.h"
#include "internal.h"
#include "tapp/error.h"
#include "tapp/status.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <omp.h>

void worker_loop(executor* exec) {
    omp_set_dynamic(0);
    while (true) {
        struct task task;

        {
            std::unique_lock<std::mutex> lk(exec->m);
            exec->cv.wait(lk, [exec]{return !exec->tasks.empty() || exec->stopping;});
            if (exec->stopping && exec->tasks.empty()) break;
            task = std::move(exec->tasks.front());
            exec->tasks.pop();
        }
        // An exception must not end the worker: it becomes the error of the task
        TAPP_error error;
        try
        {
            error = task.func();
        }
        catch (const std::bad_alloc&)
        {
            error = TAPP_TBLIS_ERROR_OUT_OF_MEMORY;
        }
        catch (...)
        {
            error = TAPP_TBLIS_ERROR_INTERNAL;
        }
        if (task.error != nullptr) *task.error = error;

        {
            std::lock_guard<std::mutex> lk(exec->m);
            exec->completed++;
        }
        exec->done_cv.notify_all();
    }
}

TAPP_error submit_task(executor* exec, TAPP_status* status, task_function func) {
    struct status* status_ptr = nullptr;
    // The allocations may throw, which must not leave this C interface. Nothing is queued then and no status is given out
    try
    {
        struct task task;
        task.func = std::move(func);
        if (status != nullptr)
        {
            status_ptr = new struct status;
            status_ptr->exec = (TAPP_executor)exec;
            // The worker replaces it with the error of the task once the task has run
            status_ptr->error = std::make_shared<std::atomic<TAPP_error>>(TAPP_TBLIS_STATUS_RUNNING);
            task.error = status_ptr->error;
        }

        std::lock_guard<std::mutex> lk(exec->m);
        exec->tasks.push(std::move(task));
        uint64_t id = ++exec->submitted;
        if (status_ptr != nullptr) status_ptr->task_id = id;
    }
    catch (const std::bad_alloc&)
    {
        delete status_ptr;
        return TAPP_TBLIS_ERROR_OUT_OF_MEMORY;
    }

    if (status != nullptr) *status = (TAPP_status)status_ptr;
    exec->cv.notify_one();
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_create_executor(TAPP_executor* exec, TAPP_attr attr) {
    if (!exec) return TAPP_TBLIS_ERROR_NULL_POINTER;

    bool has_cpu_mask = false;
    unsigned thread_count = tblis_get_num_threads();
#ifdef TAPP_TBLIS_HAS_CPU_MASK
    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);
#endif

    // Without attributes the executor keeps the defaults above
    if (attr != 0)
    {
        const attributes& a = *reinterpret_cast<attributes*>(attr);

#ifdef TAPP_TBLIS_HAS_CPU_MASK
        std::map<TAPP_key, std::vector<char>>::const_iterator cpu_mask_itr = a.find(TAPP_CPU_MASK);
        if (cpu_mask_itr != a.end())
        {
            cpu_set_t n;
            std::memcpy(&n, cpu_mask_itr->second.data(), sizeof(cpu_set_t));
            cpu_mask = n;
            has_cpu_mask = true;
            thread_count = CPU_COUNT(&cpu_mask);
        }
#endif

        std::map<TAPP_key, std::vector<char>>::const_iterator thread_count_itr = a.find(TAPP_THREAD_COUNT);
        if (thread_count_itr != a.end())
        {
            unsigned n;
            std::memcpy(&n, thread_count_itr->second.data(), sizeof(unsigned));
            thread_count = n;
        }
    }

    executor* backend_exec = new executor();
#ifdef TAPP_TBLIS_HAS_CPU_MASK
    backend_exec->cpu_mask = cpu_mask;
#endif
    backend_exec->has_cpu_mask = has_cpu_mask;
    backend_exec->thread_count = thread_count;

    backend_exec->worker = std::thread(worker_loop, backend_exec);

#ifdef TAPP_TBLIS_HAS_CPU_MASK
    if (has_cpu_mask)
    {
        // The kernel only refuses a mask without any usable CPU, from any other it silently drops the CPUs that are
        // not usable. Read back what the worker got: the executor must have all the CPUs it asked for
        cpu_set_t granted;
        int err = pthread_setaffinity_np(backend_exec->worker.native_handle(), sizeof(cpu_set_t), &backend_exec->cpu_mask);
        if (err == 0) err = pthread_getaffinity_np(backend_exec->worker.native_handle(), sizeof(cpu_set_t), &granted);
        if (err != 0 || !CPU_EQUAL(&granted, &backend_exec->cpu_mask))
        {
            TAPP_destroy_executor((TAPP_executor)backend_exec);
            return TAPP_TBLIS_ERROR_CPU_MASK_UNAVAILABLE;
        }
    }
#endif

    *exec = (TAPP_executor) backend_exec;
    return 0;
}

TAPP_error TAPP_destroy_executor(TAPP_executor exec) {
    if (exec == 0) return TAPP_TBLIS_SUCCESS;

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

TAPP_error TAPP_wait_all(TAPP_executor exec) {
    if (!exec) return TAPP_TBLIS_ERROR_NULL_POINTER;

    executor* backend_exec = (executor*)exec;
    std::unique_lock<std::mutex> lk(backend_exec->m);
    backend_exec->done_cv.wait(lk, [backend_exec]{ return backend_exec->completed == backend_exec->submitted;});
    return 0;
}

TAPP_error TAPP_wait_one(TAPP_executor exec, TAPP_status status) {
    if (!exec || !status) return TAPP_TBLIS_ERROR_NULL_POINTER;
    if (exec != ((struct status*)status)->exec) return TAPP_TBLIS_ERROR_STATUS_WRONG_EXECUTOR;

    executor* backend_exec = (executor*)exec;
    std::unique_lock<std::mutex> lk(backend_exec->m);
    // A task_id is the number of its task on this executor, starting at 1, set by submit_task under the lock. Only a
    // status that is corrupted or already destroyed fails the check
    uint64_t task_id = ((struct status*)status)->task_id;
    if (task_id <= 0 || (uint64_t)task_id > backend_exec->submitted) return TAPP_TBLIS_ERROR_INVALID_STATUS;
    backend_exec->done_cv.wait(lk, [backend_exec, task_id]{return backend_exec->completed >= task_id;});
    return 0;
}