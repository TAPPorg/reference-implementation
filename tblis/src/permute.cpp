#include "internal.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <omp.h>

static TAPP_error check_permute_indices(const tensor_info* info_X, const int64_t* idx_X, const tensor_info* info_Y, const int64_t* idx_Y) {
    // A permute only reorders: X and Y have the same indices, each exactly once, with the same extents
    std::unordered_map<int64_t, int64_t> extent_of;
    // Marks an index of X that Y has used, extents are never negative
    constexpr int64_t consumed = -1;
    auto record = [&extent_of](const tensor_info* info, const int64_t* idx) {
        for (size_t i = 0; i < info->extents.size(); i++)
        {
            if (!extent_of.emplace(idx[i], info->extents[i]).second) return TAPP_TBLIS_ERROR_REPEATED_INDEX;
        }
        return TAPP_TBLIS_SUCCESS;
    };
    auto consume = [&extent_of](const tensor_info* info, const int64_t* idx) {
        for (size_t i = 0; i < info->extents.size(); i++)
        {
            auto it = extent_of.find(idx[i]);
            if (it == extent_of.end()) return TAPP_TBLIS_ERROR_INDEX_NOT_IN_INPUTS;
            if (it->second == consumed) return TAPP_TBLIS_ERROR_REPEATED_INDEX;
            if (it->second != info->extents[i]) return TAPP_TBLIS_ERROR_EXTENT_MISMATCH;
            it->second = consumed;
        }
        // Y has no repeats and only indices of X, so it has used all of them exactly when it has as many
        if (info->extents.size() != extent_of.size()) return TAPP_TBLIS_ERROR_INDEX_NOT_IN_OUTPUT;
        return TAPP_TBLIS_SUCCESS;
    };
    TAPP_error error = record(info_X, idx_X);
    if (error != TAPP_TBLIS_SUCCESS) return error;
    return consume(info_Y, idx_Y);
}

TAPP_error TAPP_create_tensor_permute(TAPP_tensor_permute* plan,
                                      TAPP_element_op op_A,
                                      TAPP_tensor_info A,
                                      const int64_t* idx_A,
                                      TAPP_element_op op_B,
                                      TAPP_tensor_info B,
                                      const int64_t* idx_B,
                                      TAPP_element_op op_C,
                                      TAPP_tensor_info C,
                                      const int64_t* idx_C,
                                      TAPP_prectype prec) {
    const tensor_info* info_A = reinterpret_cast<tensor_info*>(A);
    const tensor_info* info_B = reinterpret_cast<tensor_info*>(B);
    const tensor_info* info_C = reinterpret_cast<tensor_info*>(C);
    if (!plan || !info_A || !info_B || !info_C) return TAPP_TBLIS_ERROR_NULL_POINTER;
    if ((!info_A->extents.empty() && !idx_A) || (!info_B->extents.empty() && !idx_B) ||
        (!info_C->extents.empty() && !idx_C))
    {
        return TAPP_TBLIS_ERROR_NULL_POINTER;
    }

    if (info_A->type != info_C->type || info_B->type != info_C->type)
    {
        return TAPP_TBLIS_ERROR_DATATYPE_MISMATCH;
    }

    if (!same_structure(info_B, idx_B, info_C, idx_C))
    {
        // C is computed in the memory of B, so they must have the same structure
        return TAPP_TBLIS_ERROR_B_C_MISMATCH;
    }

    try
    {
        TAPP_error index_check_error = check_permute_indices(info_A, idx_A, info_C, idx_C);
        if (index_check_error != TAPP_TBLIS_SUCCESS) return index_check_error;

        std::shared_ptr<permute_plan> p = std::make_shared<permute_plan>();

        p->A = *info_A;
        p->B = *info_B;
        p->C = *info_C;
        bool flip = (op_C == TAPP_CONJUGATE);
        p->A.conj = (op_A == TAPP_CONJUGATE) != flip;
        p->B.conj = (op_B == TAPP_CONJUGATE) != flip;
        p->C.conj = false;
        p->conj_scalars = flip;
        
        std::unordered_map<int64_t, tblis::label_type> label_map;
        if (!translate_idx_to_tblis(p->A.extents.size(), idx_A, label_map, p->idx_A) ||
            !translate_idx_to_tblis(p->B.extents.size(), idx_B, label_map, p->idx_B))
        {
            return TAPP_TBLIS_ERROR_TOO_MANY_INDICES;
        }
        *plan = reinterpret_cast<TAPP_tensor_permute>(new std::shared_ptr<const permute_plan>(std::move(p)));
    }
    catch (const std::bad_alloc&)
    {
        return TAPP_TBLIS_ERROR_OUT_OF_MEMORY;
    }
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_destroy_tensor_permute(TAPP_tensor_permute plan) {
    delete reinterpret_cast<std::shared_ptr<const permute_plan>*>(plan);
    return TAPP_TBLIS_SUCCESS;
}
 
TAPP_error TAPP_execute_permute(TAPP_tensor_permute plan,
                                TAPP_executor exec,
                                TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* beta,
                                const void* B,
                                      void* C) {
    if (!plan || !exec || !alpha || !beta || !A || !C) return TAPP_TBLIS_ERROR_NULL_POINTER;
    
    executor* e = reinterpret_cast<executor*>(exec);
    // A copy of the handle's shared pointer: the permute keeps the plan alive until it has run
    std::shared_ptr<const permute_plan> p = *reinterpret_cast<std::shared_ptr<const permute_plan>*>(plan);

    // Read alpha and beta NOW: they point into the caller's memory, which may be gone when the task runs
    tblis::tblis_scalar tblis_alpha = to_tblis_scalar(p->C.type, alpha);
    tblis::tblis_scalar tblis_beta = to_tblis_scalar(p->C.type, beta);
    if (!B && !tblis_beta.is_zero()) return TAPP_TBLIS_ERROR_NULL_POINTER;
    if (p->conj_scalars)
    {
        tblis_alpha.conj();
        tblis_beta.conj();
    }

    uint64_t id = submit_task(e, [=]() {
        return execute_permute(p.get(), e, tblis_alpha, A, tblis_beta, B, C);
    });
    
    if (status) *status = static_cast<TAPP_status>(id);
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error execute_permute(const permute_plan* plan,
                           const executor* exec,
                           const tblis::scalar alpha,
                           const void* A,
                           const tblis::scalar beta,
                           const void* B,
                                 void* C) {
    // C is computed in two steps: C = beta * op(B), then C += alpha * op(A). The copy follows the strides of B and C, so
    // any layout works. op(B) is applied as the source of the copy and C is never a conjugated output: tblis_tensor_add
    // gets a conjugated output wrong when both tensors have the same layout, it calls the scalv kernel with the
    // arguments of scal2v (TBLIS 9b95712, frame/1t/dense/add.cxx, TBLIS issue #87)
    const tblis::tblis_scalar zero(0.0, plan->C.type);

    tblis::tblis_tensor tblis_A = create_tblis_tensor(plan->A, A, &alpha);
    tblis::tblis_tensor tblis_B = create_tblis_tensor(plan->B, B, &beta);
    tblis::tblis_tensor tblis_C = create_tblis_tensor(plan->C, C, &zero);

    // OpenMP may give fewer threads than asked for (OMP_THREAD_LIMIT). TCI then lets every thread run the whole
    // permute on the same C, so ask first and only use what is there
    unsigned thread_count = exec->thread_count;
    unsigned omp_thread_count = thread_count;
    #pragma omp parallel num_threads(thread_count)
    {
        #pragma omp single
        omp_thread_count = omp_get_num_threads();
    }
    if (omp_thread_count < thread_count)
    {
        thread_count = omp_thread_count;
    }
    tblis::parallelize([&](const tblis::communicator& comm)
    {
        // With binding on (OMP_PROC_BIND, OMP_PLACES), OpenMP pins its threads itself, so they do not inherit the mask
        // from the worker: every thread puts itself back on it
#ifdef TAPP_TBLIS_HAS_CPU_MASK
        if (exec->has_cpu_mask && omp_get_proc_bind() != omp_proc_bind_false)
        {
            sched_setaffinity(0, sizeof(cpu_set_t), &exec->cpu_mask);
        }
#endif

        // TBLIS resets scalar and conj of an output tensor when it is done with it, so every thread needs its own
        tblis::tblis_tensor tC_thread = tblis_C;
        // With beta == 0 there is nothing to copy, and the add below keeps the scalar 0 of C so that it only writes C.
        // This also keeps a conjugated A on the correct beta == 0 branch of TBLIS's transpose kernel (see below)
        if (!beta.is_zero())
        {
            // C = beta * op(B). tblis_tensor_add resets the scalar of its output to 1 when it is done, so the add below
            // adds A to this result
            tblis::tblis_tensor_add(comm, nullptr, &tblis_B, plan->idx_B.data(), &tC_thread, plan->idx_B.data());
            // Every thread's part of C must be written before anyone adds to it
            comm.barrier();
        }

        // Known issue: with a conjugated A, beta != 0 and a change of index order, the result is wrong. TBLIS (9b95712)
        // then uses its transpose kernel, whose beta != 0 branch lacks the elses between its cases
        // (plugin/ref_kernels/trans_ref.cxx, TBLIS issue #88), so with conj_A a block is computed two or three times.
        // Possible workaround if needed: compute conj(C) = conj(alpha) * A + conj(beta) * conj(B) instead, so A is not
        // conjugated (conj on B alone takes a correct branch), then conjugate C in place with tblis_tensor_scale.
        tblis::tblis_tensor_add(comm, nullptr,
                                &tblis_A, plan->idx_A.data(),
                                &tC_thread, plan->idx_B.data());
    }, thread_count);

    return TAPP_TBLIS_SUCCESS;
}