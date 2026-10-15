/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "internal.h"
#include "tapp/error.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <omp.h>

TAPP_error TAPP_create_tensor_product(TAPP_tensor_product* plan,
                                      TAPP_executor exec,
                                      TAPP_element_op op_A,
                                      TAPP_tensor_info A,
                                      const int64_t* idx_A,
                                      TAPP_element_op op_B,
                                      TAPP_tensor_info B,
                                      const int64_t* idx_B,
                                      TAPP_element_op op_C,
                                      TAPP_tensor_info C,
                                      const int64_t* idx_C,
                                      TAPP_element_op op_D,
                                      TAPP_tensor_info D,
                                      const int64_t* idx_D,
                                      TAPP_prectype prec) {
    const tensor_info* info_A = reinterpret_cast<tensor_info*>(A);
    const tensor_info* info_B = reinterpret_cast<tensor_info*>(B);
    const tensor_info* info_C = reinterpret_cast<tensor_info*>(C);
    const tensor_info* info_D = reinterpret_cast<tensor_info*>(D);
    if (!plan || !info_A || !info_B || !info_C || !info_D) return TAPP_TBLIS_ERROR_NULL_POINTER;
    if ((!info_A->extents.empty() && !idx_A) || (!info_B->extents.empty() && !idx_B) ||
        (!info_C->extents.empty() && !idx_C) || (!info_D->extents.empty() && !idx_D))
    {
        return TAPP_TBLIS_ERROR_NULL_POINTER;
    }

    // TBLIS multiplies tensors of one type, and alpha and beta are read as that type
    if (info_A->type != info_D->type || info_B->type != info_D->type || info_C->type != info_D->type)
    {
        return TAPP_TBLIS_ERROR_DATATYPE_MISMATCH;
    }

    if (!same_structure(info_C, idx_C, info_D, idx_D))
    {
        // Tensor C and D does not have the same structure
        return TAPP_TBLIS_ERROR_C_D_MISMATCH;
    }

    try
    {
        // An index is one loop: it has the same extent wherever it is used, and D only has indices of A and B.
        // C needs no pass of its own, it has the structure of D
        std::unordered_map<int64_t, int64_t> extent_of;
        auto add = [&extent_of](const tensor_info* info, const int64_t* idx) {
            for (size_t i = 0; i < info->extents.size(); i++)
            {
                auto [it, is_new] = extent_of.emplace(idx[i], info->extents[i]);
                if (!is_new && it->second != info->extents[i]) return false;
            }
            return true;
        };
        if (!add(info_A, idx_A) || !add(info_B, idx_B)) return TAPP_TBLIS_ERROR_EXTENT_MISMATCH;
        for (size_t i = 0; i < info_D->extents.size(); i++)
        {
            auto it = extent_of.find(idx_D[i]);
            if (it == extent_of.end()) return TAPP_TBLIS_ERROR_INDEX_NOT_IN_INPUTS;
            if (it->second != info_D->extents[i]) return TAPP_TBLIS_ERROR_EXTENT_MISMATCH;
        }

        std::shared_ptr<product_plan> p = std::make_shared<product_plan>();

        p->A = *info_A;
        p->B = *info_B;
        p->C = *info_C;
        p->D = *info_D;
        bool flip = (op_D == TAPP_CONJUGATE);
        p->A.conj = (op_A == TAPP_CONJUGATE) != flip;
        p->B.conj = (op_B == TAPP_CONJUGATE) != flip;
        p->C.conj = (op_C == TAPP_CONJUGATE) != flip;
        p->D.conj = false;
        p->conj_scalars = flip;

        std::unordered_map<int64_t, tblis::label_type> label_map;
        if (!translate_idx_to_tblis(p->A.extents.size(), idx_A, label_map, p->idx_A) ||
            !translate_idx_to_tblis(p->B.extents.size(), idx_B, label_map, p->idx_B) ||
            !translate_idx_to_tblis(p->C.extents.size(), idx_C, label_map, p->idx_C))
        {
            return TAPP_TBLIS_ERROR_TOO_MANY_INDICES;
        }
        *plan = reinterpret_cast<TAPP_tensor_product>(new std::shared_ptr<const product_plan>(std::move(p)));
    }
    catch (const std::bad_alloc&)
    {
        return TAPP_TBLIS_ERROR_OUT_OF_MEMORY;
    }
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan) {
    // Only the handle's share of the plan: products that are queued or running keep it alive until they are done
    delete reinterpret_cast<std::shared_ptr<const product_plan>*>(plan);
    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_execute_product(TAPP_tensor_product plan,
                                TAPP_executor exec,
                                TAPP_status* status,
                                const void* alpha,
                                const void* A,
                                const void* B,
                                const void* beta,
                                const void* C,
                                      void* D) {
    if (!plan || !exec || !alpha || !beta || !A || !B || !D) return TAPP_TBLIS_ERROR_NULL_POINTER;
    
    executor* e = reinterpret_cast<executor*>(exec);
    // A copy of the handle's shared pointer: the product keeps the plan alive until it has run
    std::shared_ptr<const product_plan> p = *reinterpret_cast<std::shared_ptr<const product_plan>*>(plan);

    // Read alpha and beta NOW: they point into the caller's memory, which may be gone when the task runs
    tblis::tblis_scalar tblis_alpha = to_tblis_scalar(p->D.type, alpha);
    tblis::tblis_scalar tblis_beta = to_tblis_scalar(p->D.type, beta);
    if (!C && !tblis_beta.is_zero()) return TAPP_TBLIS_ERROR_NULL_POINTER;
    if (p->conj_scalars)
    {
        tblis_alpha.conj();
        tblis_beta.conj();
    }

    TAPP_error error = submit_task(e, status, [=]() {
        return execute_product(p.get(), e, tblis_alpha, A, B, tblis_beta, C, D);
    });
    
    return error;
}

TAPP_error execute_product(const product_plan* plan,
                           const executor* exec,
                           const tblis::tblis_scalar alpha,
                           const void* A,
                           const void* B,
                           const tblis::tblis_scalar beta,
                           const void* C,
                                 void* D)
{    
    // D is computed in two steps: D = beta * op(C), then D += alpha * op(A) * op(B). The copy follows the strides of C
    // and D, so any layout works. op(C) is applied as the source of the copy and D is never a conjugated output:
    // tblis_tensor_mult conjugates a conjugated output in place using the strides of B for the indices B and the output
    // share (TBLIS 9b95712, gemm_bsmtc_blis in frame/3t/dense/mult.cxx, TBLIS issue #91). When those strides differ,
    // e.g. for D_ji = A_ik * B_kj, it conjugates the wrong elements and writes outside of D
    const tblis::tblis_scalar zero(0.0, plan->D.type);

    tblis::tblis_tensor tblis_A = create_tblis_tensor(plan->A, A, &alpha);
    tblis::tblis_tensor tblis_B = create_tblis_tensor(plan->B, B, nullptr);
    tblis::tblis_tensor tblis_C = create_tblis_tensor(plan->C, C, &beta);
    tblis::tblis_tensor tblis_D = create_tblis_tensor(plan->D, D, &zero);

    // OpenMP may give fewer threads than asked for (OMP_THREAD_LIMIT). TCI then lets every thread run the whole
    // product on the same D, so ask first and only use what is there
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
        tblis::tblis_tensor tD_thread = tblis_D;
        // With beta == 0 there is nothing to copy, and the product below keeps the scalar 0 of D so that it only
        // writes D
        if (!beta.is_zero())
        {
            // D = beta * op(C). tblis_tensor_add resets the scalar of its output to 1 when it is done, so the product
            // below adds to this result
            tblis::tblis_tensor_add(comm, nullptr,
                                    &tblis_C, plan->idx_C.data(),
                                    &tD_thread, plan->idx_C.data());
            // Every thread's part of D must be written before the product reads it
            comm.barrier();
        }

        tblis::tblis_tensor_mult(comm, nullptr,
                                 &tblis_A, plan->idx_A.data(),
                                 &tblis_B, plan->idx_B.data(),
                                 &tD_thread, plan->idx_C.data());
    }, thread_count);

    return TAPP_TBLIS_SUCCESS;
}

TAPP_error TAPP_execute_batched_product(TAPP_tensor_product plan,
                                        TAPP_executor exec,
                                        TAPP_status* status,
                                        int num_batches,
                                        const void* alpha,
                                        const void** A,
                                        const void** B,
                                        const void* beta,
                                        const void** C,
                                              void** D) {
    // TODO: loop over batches calling TAPP_execute_product
    return 0;
}
