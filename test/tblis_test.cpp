/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */

// Checks the TBLIS-backed implementation (tapp-tblis) against a naive loop implementation of
// D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))
#include <tapp.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <random>
#include <vector>

using cd = std::complex<double>;

struct T
{
    std::vector<int64_t> ext, str, idx;
    int64_t span() const { int64_t s = 1; for (size_t i = 0; i < ext.size(); i++) s += (ext[i] - 1) * str[i]; return s; }
};

static T colmajor(std::vector<int64_t> ext, std::vector<int64_t> idx)
{
    T t{ext, {}, idx};
    int64_t s = 1;
    for (auto e : ext) { t.str.push_back(s); s *= e; }
    return t;
}

static T rowmajor(std::vector<int64_t> ext, std::vector<int64_t> idx)
{
    T t{ext, std::vector<int64_t>(ext.size()), idx};
    int64_t s = 1;
    for (int i = (int)ext.size() - 1; i >= 0; i--) { t.str[i] = s; s *= ext[i]; }
    return t;
}

template <typename V> static V cj(V v) { return v; }
template <> cd cj<cd>(cd v) { return std::conj(v); }

template <typename V>
static void reference(const T& a, const std::vector<V>& A, int opA,
                      const T& b, const std::vector<V>& B, int opB,
                      const T& c, const std::vector<V>* C, int opC,
                      const T& d, std::vector<V>& D, int opD, V alpha, V beta)
{
    std::map<int64_t, int64_t> ext;
    for (const T* t : {&a, &b, &d}) for (size_t i = 0; i < t->idx.size(); i++) ext[t->idx[i]] = t->ext[i];
    std::vector<int64_t> labels; for (auto& [l, e] : ext) labels.push_back(l);

    auto off = [&](const T& t, std::map<int64_t, int64_t>& pos) {
        int64_t o = 0; for (size_t i = 0; i < t.idx.size(); i++) o += pos[t.idx[i]] * t.str[i]; return o;
    };

    std::vector<V> acc(D.size(), V(0));
    std::vector<bool> touched(D.size(), false);
    std::map<int64_t, int64_t> pos; for (auto l : labels) pos[l] = 0;
    bool empty = false; for (auto& [l, e] : ext) if (e == 0) empty = true;

    // beta * op_C(C) for every element of D
    {
        std::map<int64_t, int64_t> p; for (auto l : d.idx) p[l] = 0;
        int64_t n = 1; for (auto e : d.ext) n *= e;
        for (int64_t k = 0; k < n; k++)
        {
            int64_t r = k; for (size_t i = 0; i < d.idx.size(); i++) { p[d.idx[i]] = r % d.ext[i]; r /= d.ext[i]; }
            int64_t od = off(d, p);
            acc[od] = (C && beta != V(0)) ? beta * (opC ? cj((*C)[off(c, p)]) : (*C)[off(c, p)]) : V(0);
            touched[od] = true;
        }
    }
    if (!empty)
    {
        while (true)
        {
            V va = A[off(a, pos)], vb = B[off(b, pos)];
            acc[off(d, pos)] += alpha * (opA ? cj(va) : va) * (opB ? cj(vb) : vb);
            size_t i = 0;
            for (; i < labels.size(); i++) { if (++pos[labels[i]] < ext[labels[i]]) break; pos[labels[i]] = 0; }
            if (i == labels.size()) break;
        }
    }
    for (size_t i = 0; i < D.size(); i++) if (touched[i]) D[i] = opD ? cj(acc[i]) : acc[i];
}

template <typename V> static TAPP_datatype dtype();
template <> TAPP_datatype dtype<float>() { return TAPP_F32; }
template <> TAPP_datatype dtype<double>() { return TAPP_F64; }
template <> TAPP_datatype dtype<cd>() { return TAPP_C64; }

template <typename V> static V rnd(std::mt19937& g);
template <> float rnd<float>(std::mt19937& g) { return std::uniform_real_distribution<float>(-1, 1)(g); }
template <> double rnd<double>(std::mt19937& g) { return std::uniform_real_distribution<double>(-1, 1)(g); }
template <> cd rnd<cd>(std::mt19937& g) { std::uniform_real_distribution<double> u(-1, 1); return {u(g), u(g)}; }

static TAPP_tensor_info info(const T& t, TAPP_datatype type)
{
    TAPP_tensor_info i;
    TAPP_error e = TAPP_create_tensor_info(&i, type, (int)t.ext.size(), t.ext.data(), t.str.data());
    if (e != 0) { std::printf("  create_tensor_info failed: %d\n", e); std::exit(1); }
    return i;
}

static int failures = 0;

// in_place: pass D as C. c_null: pass NULL for C (beta must be 0). nan_D: prefill D with NaN.
template <typename V>
static void run(const char* name, const T& a, const T& b, const T& d,
                int opA, int opB, int opC, int opD, V alpha, V beta,
                bool in_place = false, bool c_null = false, bool nan_D = false)
{
    std::mt19937 g(42);
    std::vector<V> A(a.span()), B(b.span()), C(d.span()), D(d.span());
    for (auto& x : A) x = rnd<V>(g);
    for (auto& x : B) x = rnd<V>(g);
    for (auto& x : C) x = rnd<V>(g);
    for (auto& x : D) x = nan_D ? V(std::numeric_limits<double>::quiet_NaN()) : rnd<V>(g);
    if (in_place) D = C;

    std::vector<V> R = D;
    reference(a, A, opA, b, B, opB, d, (c_null ? nullptr : &C), opC, d, R, opD, alpha, beta);
    if (in_place) { std::vector<V> Cc = D; R = D; reference(a, A, opA, b, B, opB, d, &Cc, opC, d, R, opD, alpha, beta); }

    TAPP_executor ex; TAPP_create_executor(&ex, 0);
    TAPP_tensor_info ia = info(a, dtype<V>()), ib = info(b, dtype<V>()), ic = info(d, dtype<V>()), id = info(d, dtype<V>());
    TAPP_tensor_product plan;
    TAPP_error e = TAPP_create_tensor_product(&plan, opA, ia, a.idx.data(), opB, ib, b.idx.data(),
                                              opC, ic, d.idx.data(), opD, id, d.idx.data(), TAPP_DEFAULT_PREC);
    if (e != 0) { std::printf("FAIL %-45s create_tensor_product returned %d\n", name, e); failures++; return; }

    // Destroy the user's tensor infos right away: the plan must not depend on them
    TAPP_destroy_tensor_info(ia); TAPP_destroy_tensor_info(ib); TAPP_destroy_tensor_info(ic); TAPP_destroy_tensor_info(id);

    TAPP_status st = 0;
    {
        V alpha_local = alpha, beta_local = beta;   // goes out of scope before the wait, like a caller's locals would
        const void* Cp = c_null ? nullptr : (in_place ? (const void*)D.data() : (const void*)C.data());
        e = TAPP_execute_product(plan, ex, &st, &alpha_local, A.data(), B.data(), &beta_local, Cp, D.data());
        alpha_local = V(999); beta_local = V(999);
    }
    if (e != 0) { std::printf("FAIL %-45s execute_product returned %d\n", name, e); failures++; return; }
    TAPP_wait_one(ex, st);

    double err = 0, scale = 0;
    bool nan = false;
    for (size_t i = 0; i < D.size(); i++)
    {
        if (std::isnan(std::abs(D[i]))) nan = true;
        err = std::max(err, (double)std::abs(D[i] - R[i]));
        scale = std::max(scale, (double)std::abs(R[i]));
    }
    double tol = (sizeof(V) == sizeof(float) ? 1e-5 : 1e-12) * std::max(1.0, scale);
    bool ok = !nan && err <= tol;
    std::printf("%s %-45s max error %.2e%s\n", ok ? "ok  " : "FAIL", name, err, nan ? " (NaN in D)" : "");
    if (!ok) failures++;

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_executor(ex);
}

int main()
{
    enum { I = TAPP_IDENTITY, CJ = TAPP_CONJUGATE };
    const int64_t i = 1, j = 2, k = 3, l = 4;

    auto a = colmajor({5, 4}, {i, k}), b = colmajor({4, 3}, {k, j}), d = colmajor({5, 3}, {i, j});
    run<double>("f64 gemm, C != D", a, b, d, I, I, I, I, 1.5, 0.5);
    run<float>("f32 gemm, C != D", a, b, d, I, I, I, I, 1.5f, 0.5f);
    run<double>("f64 gemm, in place (C == D)", a, b, d, I, I, I, I, 1.5, 0.5, true);
    run<double>("f64 gemm, beta = 0, C = NULL", a, b, d, I, I, I, I, 2.0, 0.0, false, true);
    run<double>("f64 gemm, beta = 0, D prefilled with NaN", a, b, d, I, I, I, I, 2.0, 0.0, false, true, true);

    cd alpha(0.7, -1.3), beta(-0.4, 0.9);
    run<cd>("c64 gemm, no ops", a, b, d, I, I, I, I, alpha, beta);
    run<cd>("c64 gemm, conj A", a, b, d, CJ, I, I, I, alpha, beta);
    run<cd>("c64 gemm, conj B", a, b, d, I, CJ, I, I, alpha, beta);
    run<cd>("c64 gemm, conj C", a, b, d, I, I, CJ, I, alpha, beta);
    run<cd>("c64 gemm, conj C, in place", a, b, d, I, I, CJ, I, alpha, beta, true);
    run<cd>("c64 gemm, conj C, beta = 1", a, b, d, I, I, CJ, I, alpha, cd(1, 0));
    run<cd>("c64 outer product, conj C", colmajor({3}, {i}), colmajor({4}, {j}), colmajor({3, 4}, {i, j}), I, I, CJ, I, alpha, beta);
    run<cd>("c64 gemm, conj B and C", a, b, d, I, CJ, CJ, I, alpha, beta);
    run<cd>("c64 gemm, conj D", a, b, d, I, I, I, CJ, alpha, beta);
    run<cd>("c64 gemm, conj all", a, b, d, CJ, CJ, CJ, CJ, alpha, beta);
    run<cd>("c64 gemm, conj A and D, in place", a, b, d, CJ, I, I, CJ, alpha, beta, true);

    // j: Hadamard (A, B, D), k: contracted, i/l: free; D row-major, A/B col-major
    auto a3 = colmajor({3, 4, 5}, {i, j, k}), b3 = colmajor({4, 5, 2}, {j, k, l}), d3 = rowmajor({3, 4, 2}, {i, j, l});
    run<double>("f64 hadamard + contraction, mixed layouts", a3, b3, d3, I, I, I, I, 1.0, 1.0);
    run<cd>("c64 hadamard + contraction, conj B and D", a3, b3, d3, I, CJ, I, CJ, alpha, beta);

    // Dot product into a scalar
    auto av = colmajor({7}, {k}), bv = colmajor({7}, {k}), ds = colmajor({}, {});
    run<double>("f64 dot product (0-mode D)", av, bv, ds, I, I, I, I, 1.0, 1.0);

    // Outer product, no contracted index
    auto ao = colmajor({3}, {i}), bo = colmajor({4}, {j}), dout = colmajor({3, 4}, {i, j});
    run<double>("f64 outer product", ao, bo, dout, I, I, I, I, 1.0, 0.5);

    std::printf("\n%d failure(s)\n", failures);
    return failures != 0;
}
