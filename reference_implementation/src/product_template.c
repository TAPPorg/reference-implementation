/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - August 2026
 */
#include "../include/product.h"
#include "../include/product_template.h"

#define CHECK_TENSOR_EXISTENCE2(T_A, T_B, T_C, T_D) check_tensor_existence_ ## T_A ## _ ## T_B ## _ ## T_C ## _ ## T_D
#define CHECK_TENSOR_EXISTENCE(T_A, T_B, T_C, T_D) CHECK_TENSOR_EXISTENCE2(T_A, T_B, T_C, T_D)

// Conjugation only makes sense for complex values; for every real (incl. extended,
// e.g. _Float16/__bf16) storage type it is a no-op, which the default case gives us.
#define TAPP_CONJ(x) _Generic((x), \
    float complex: conjf(x), \
    double complex: conj(x), \
    default: (x))

static int CHECK_TENSOR_EXISTENCE(T_A, T_B, T_C, T_D)(const T_D scalar, const void* tensor, int error_code);

TAPP_error TAPP_EXECUTE_PRODUCT_TEMPLATE(T_A, T_B, T_C, T_D)(TAPP_tensor_product plan,
                                         TAPP_executor exec,
                                         TAPP_status* status,
                                         const void* alpha_,
                                         const void* A_,
                                         const void* B_,
                                         const void* beta_,
                                         const void* C_,
                                         void* D_)
{
    struct plan* plan_ptr = (struct plan*)plan;

    const T_D alpha = *(const T_D*)alpha_;
    const T_A* A = (const T_A*)A_;
    const T_B* B = (const T_B*)B_;
    const T_D beta = *(const T_D*)beta_;
    const T_C* C = (const T_C*)C_;
    T_D* D = (T_D*)D_;

    int64_t* H_coords = checked_malloc(plan_ptr->H_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->H_nmode; i++) H_coords[i] = 0;

    int64_t* P_coords = checked_malloc(plan_ptr->P_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->P_nmode; i++) P_coords[i] = 0;

    int64_t* FA_coords = checked_malloc(plan_ptr->FA_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->FA_nmode; i++) FA_coords[i] = 0;

    int64_t* FB_coords = checked_malloc(plan_ptr->FB_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->FB_nmode; i++) FB_coords[i] = 0;

    int64_t* IA_coords = checked_malloc(plan_ptr->IA_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->IA_nmode; i++) IA_coords[i] = 0;

    int64_t* IB_coords = checked_malloc(plan_ptr->IB_nmode * sizeof(int64_t));
    for (int i = 0; i < plan_ptr->IB_nmode; i++) IB_coords[i] = 0;

    int error_status = 0;

    if (error_status == 0) error_status = CHECK_TENSOR_EXISTENCE(T_A, T_B, T_C, T_D)(beta, C, 12);
    if (error_status == 0) error_status = check_executor_existence(exec, 33);
    if (error_status != 0)
    {
        return error_status;
    }

    for (int64_t h = 0; h < plan_ptr->H_size; h++)
    {
        int64_t H_offset_A = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_A);
        int64_t H_offset_B = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_B);
        int64_t H_offset_D = calcualte_offset(H_coords, plan_ptr->H_nmode, plan_ptr->H_strides_D);

        for (int64_t fa = 0; fa < plan_ptr->FA_size; fa++)
        {
            int64_t FA_offset_A = calcualte_offset(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_strides_A);
            int64_t FA_offset_D = calcualte_offset(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_strides_D);

            for (int64_t fb = 0; fb < plan_ptr->FB_size; fb++)
            {
                T_D accum = 0;

                int64_t FB_offset_B = calcualte_offset(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_strides_B);
                int64_t FB_offset_D = calcualte_offset(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_strides_D);

                int64_t offset_D = H_offset_D + FA_offset_D + FB_offset_D;

                if (beta != 0)
                {
                    accum = beta * (T_D)(plan_ptr->op_C == TAPP_CONJUGATE ? TAPP_CONJ(C[offset_D]) : C[offset_D]);
                }

                for (int64_t p = 0; p < plan_ptr->P_size; p++)
                {
                    int64_t P_offset_A = calcualte_offset(P_coords, plan_ptr->P_nmode, plan_ptr->P_strides_A);
                    int64_t P_offset_B = calcualte_offset(P_coords, plan_ptr->P_nmode, plan_ptr->P_strides_B);

                    T_A sum_A = 0;
                    for (int64_t ia = 0; ia < plan_ptr->IA_size; ia++)
                    {
                        int64_t IA_offset_A = calcualte_offset(IA_coords, plan_ptr->IA_nmode, plan_ptr->IA_strides_A);
                        int64_t offset_A = H_offset_A + FA_offset_A + P_offset_A + IA_offset_A;
                        sum_A += A[offset_A];
                        increment_coordinates(IA_coords, plan_ptr->IA_nmode, plan_ptr->IA_extents);
                    }

                    T_B sum_B = 0;
                    for (int64_t ib = 0; ib < plan_ptr->IB_size; ib++)
                    {
                        int64_t IB_offset_B = calcualte_offset(IB_coords, plan_ptr->IB_nmode, plan_ptr->IB_strides_B);
                        int64_t offset_B = H_offset_B + FB_offset_B + P_offset_B + IB_offset_B;
                        sum_B += B[offset_B];
                        increment_coordinates(IB_coords, plan_ptr->IB_nmode, plan_ptr->IB_extents);
                    }

                    accum += alpha
                           * (T_D)(plan_ptr->op_A == TAPP_CONJUGATE ? TAPP_CONJ(sum_A) : sum_A)
                           * (T_D)(plan_ptr->op_B == TAPP_CONJUGATE ? TAPP_CONJ(sum_B) : sum_B);

                    increment_coordinates(P_coords, plan_ptr->P_nmode, plan_ptr->P_extents);
                }

                D[offset_D] = plan_ptr->op_D == TAPP_CONJUGATE ? TAPP_CONJ(accum) : accum;

                increment_coordinates(FB_coords, plan_ptr->FB_nmode, plan_ptr->FB_extents);
            }

            increment_coordinates(FA_coords, plan_ptr->FA_nmode, plan_ptr->FA_extents);
        }

        increment_coordinates(H_coords, plan_ptr->H_nmode, plan_ptr->H_extents);
    }

    free(H_coords);
    free(P_coords);
    free(FA_coords);
    free(FB_coords);
    free(IA_coords);
    free(IB_coords);

    return 0;
}

static int CHECK_TENSOR_EXISTENCE(T_A, T_B, T_C, T_D)(const T_D scalar, const void* tensor, int error_code)
{
    return tensor == NULL && scalar != (T_D)0 ? error_code : 0;
}
