/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */
#include "../include/product.h"
#include "../include/executor.h"
#include "datatype.h"
#include "tapp/error.h"
#include "tapp/executor.h"
#include "tapp/product.h"
#include "tapp/tensor.h"
#include "tblis/frame/3t/mult.h"
#include "tblis/frame/base/basic_types.h"
#include "tblis_utils.h"
#include "tensor.h"
#include <cstdint>
#include <cstdlib>
#include <error.h>
#include <unordered_map>
#include <cstring>

TAPP_error TAPP_create_tensor_product(TAPP_tensor_product* plan,
                                      TAPP_handle handle,
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
    product_plan* p = new product_plan();
    if (!same_structure(reinterpret_cast<tensor_info*>(C), idx_C, reinterpret_cast<tensor_info*>(D), idx_D))
    {
        // Tensor C and D does not have the same structure
        return ERROR_C_D_MISMATCH;
    }
    p->A = clone_tensor_info(reinterpret_cast<tensor_info*>(A));
    p->B = clone_tensor_info(reinterpret_cast<tensor_info*>(B));
    p->C = clone_tensor_info(reinterpret_cast<tensor_info*>(C));
    bool flip = (op_D == TAPP_CONJUGATE);
    p->A->conj = (op_A == TAPP_CONJUGATE) != flip;
    p->B->conj = (op_B == TAPP_CONJUGATE) != flip;
    p->C->conj = (op_C == TAPP_CONJUGATE) != flip;
    p->conj_scalars = flip;

    // Number of elements in D: the product of its extents (1 for a scalar, 0 if any extent is 0)
    auto* info_D = reinterpret_cast<tensor_info*>(D);
    p->size_D = 1;
    for (int i = 0; i < info_D->nmode; i++)
    {
        p->size_D *= info_D->extents[i];
    }
    p->scalar_type = info_D->type;

    std::unordered_map<int64_t, tblis::label_type> label_map;
    if (!translate_idx_to_tblis(p->A->nmode, idx_A, label_map, p->idx_A) ||
        !translate_idx_to_tblis(p->B->nmode, idx_B, label_map, p->idx_B) ||
        !translate_idx_to_tblis(p->C->nmode, idx_C, label_map, p->idx_C))
    {
        // too many indices
        return ERROR_TOO_MANY_INDICES;
    }
    *plan = reinterpret_cast<TAPP_tensor_product>(p);
    return 0;
}

bool same_structure(const tensor_info* A, const int64_t* idx_A, const tensor_info* B, const int64_t* idx_B) {
    if (A->nmode != B->nmode) return false;
    for (int i = 0; i < A->nmode; i++) {
        if (A->extents[i] != B->extents[i] || A->strides[i] != B->strides[i] || idx_A[i] != idx_B[i]) return false;
    }
    return true;
}

TAPP_error TAPP_destroy_tensor_product(TAPP_tensor_product plan) {
    product_plan* p = reinterpret_cast<product_plan*>(plan);
    destroy_tensor_info(p->A);
    destroy_tensor_info(p->B);
    destroy_tensor_info(p->C);
    delete p;
    return 0;
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
    executor* e = reinterpret_cast<executor*>(exec);
    const product_plan* p = reinterpret_cast<const product_plan*>(plan);

    // Read alpha and beta NOW: they point into the caller's memory, which may be gone when the task runs
    tblis::tblis_scalar alpha_s = to_tblis_scalar(p->scalar_type, alpha);
    tblis::tblis_scalar beta_s = to_tblis_scalar(p->scalar_type, beta);
    if (p->conj_scalars)
    {
        alpha_s.conj();
        beta_s.conj();
    }

    uint64_t id = submit_task(e, [=]() {
        return execute_product(p, alpha_s, A, B, beta_s, C, D);
    });

    if (status) *status = static_cast<TAPP_status>(id);
    return SUCCESS;
}

TAPP_error execute_product(const product_plan* plan,
                           const tblis::tblis_scalar alpha,
                           const void* A,
                           const void* B,
                           const tblis::tblis_scalar beta,
                           const void* C,
                                 void* D)
{
    if (C != D && !beta.is_zero())
    {
        std::memcpy(D, C, plan->size_D * tblis_type_size(plan->C->type));
    }

    tblis::tblis_tensor* tA = create_tblis_tensor(plan->A);
    tA->data = const_cast<void*>(A);
    tblis::tblis_tensor* tB = create_tblis_tensor(plan->B);
    tB->data = const_cast<void*>(B);
    tblis::tblis_tensor* tC = create_tblis_tensor(plan->C);
    tC->data = D;

    // Workaround: tblis_tensor_mult ignores conj on the output when it takes its GEMM path (TBLIS 9b95712,
    // gemm_blis in frame/3t/dense/mult.cxx), so conjugate D in place first and pass the output unconjugated
    if (tC->conj && !beta.is_zero())
    {
        tC->scalar.reset(tblis::tblis_scalar(1.0, tC->type));
        tblis::tblis_tensor_scale(nullptr, nullptr, tC, plan->idx_C.data());
    }
    tC->conj = false;

    // reset, not operator=: tblis_scalar assignment keeps the destination's type and only converts the value
    tA->scalar.reset(alpha);
    tC->scalar.reset(beta);
    

    tblis::tblis_tensor_mult(nullptr, nullptr, tA, plan->idx_A.data(), tB, plan->idx_B.data(), tC, plan->idx_C.data());

    free_tblis_tensor(tA);
    free_tblis_tensor(tB);
    free_tblis_tensor(tC);
    // TODO:
    //  1. copy C into D (unless in-place), scaled by beta
    //  2. reduce isolated indices of A and B
    //  3. tblis_tensor_mult(alpha * op_A(A), op_B(B), D)
    //  4. apply op_D
    return SUCCESS;
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
