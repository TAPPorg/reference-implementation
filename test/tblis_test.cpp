/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - October 2026
 */

// Checks the TBLIS-backed implementation (tapp-tblis) against a naive loop implementation of
// D = op_D(alpha * op_A(A) * op_B(B) + beta * op_C(C))
#include <tapp.h>
#include <tapp_tblis.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <sched.h>
#include <thread>
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

// Column-major with pad unused elements after every column, like a block of a larger matrix
static T padded(std::vector<int64_t> ext, std::vector<int64_t> idx, int64_t pad)
{
    T t{ext, {}, idx};
    int64_t s = 1;
    for (size_t i = 0; i < ext.size(); i++) { t.str.push_back(s); s *= ext[i] + (i == 0 ? pad : 0); }
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

// C = op_C(alpha * op_A(A) + beta * op_B(B)), where B has the structure of C and A has the same indices in any order.
// Elements of C outside of the tensor (padding) are left alone.
template <typename V>
static void reference_permute(const T& a, const std::vector<V>& A, int opA, const std::vector<V>* B, int opB,
                              const T& c, std::vector<V>& C, int opC, V alpha, V beta)
{
    auto off = [](const T& t, std::map<int64_t, int64_t>& pos) {
        int64_t o = 0; for (size_t i = 0; i < t.idx.size(); i++) o += pos[t.idx[i]] * t.str[i]; return o;
    };

    std::map<int64_t, int64_t> p;
    int64_t n = 1; for (auto e : c.ext) n *= e;
    for (int64_t k = 0; k < n; k++)
    {
        int64_t r = k; for (size_t i = 0; i < c.idx.size(); i++) { p[c.idx[i]] = r % c.ext[i]; r /= c.ext[i]; }
        int64_t oc = off(c, p);
        V va = A[off(a, p)];
        V v = alpha * (opA ? cj(va) : va);
        if (B && beta != V(0)) v += beta * (opB ? cj((*B)[oc]) : (*B)[oc]);
        C[oc] = opC ? cj(v) : v;
    }
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

static void check(const char* name, bool ok)
{
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name);
    if (!ok) failures++;
}

// Attributes of the executors that run() creates, 0 for the defaults
static TAPP_attr exec_attr = 0;
// Number of threads an executor of run() must have, -1 to not check. With exec_mask they must also run on that mask
static int exec_threads = -1;
static const cpu_set_t* exec_mask = nullptr;

// Number of threads of this process, or only those whose CPU mask is exactly mask
static int count_threads(const cpu_set_t* mask)
{
    int count = 0;
    for (const auto& task : std::filesystem::directory_iterator("/proc/self/task"))
    {
        cpu_set_t affinity;
        pid_t tid = std::atoi(task.path().filename().c_str());
        if (mask && (sched_getaffinity(tid, sizeof(affinity), &affinity) != 0 || !CPU_EQUAL(mask, &affinity))) continue;
        count++;
    }
    return count;
}

// The threads OpenMP started for an executor exit shortly after the executor is destroyed, not before. Wait until only
// the main thread is left, so that they are not counted as threads of the next executor
static void wait_for_exited_threads()
{
    for (int tries = 0; tries < 200 && count_threads(nullptr) > 1; tries++)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// Number of threads an executor that asks for count threads gets: OMP_THREAD_LIMIT caps it
static int limited(int count)
{
    const char* limit = std::getenv("OMP_THREAD_LIMIT");
    return (limit && std::atoi(limit) > 0) ? std::min(count, std::atoi(limit)) : count;
}

static void test_attributes()
{
    TAPP_attr attr;
    check("attr: create", TAPP_create_attr(&attr) == 0);

    size_t size = 1; void* value = &size;
    check("attr: get of unset key gives NULL",
          TAPP_attr_get(attr, TAPP_THREAD_COUNT, &size, &value) == 0 && size == 0 && value == nullptr);

    int count = 3;
    check("attr: set thread count", TAPP_attr_set(attr, TAPP_THREAD_COUNT, sizeof(count), &count) == 0);
    count = 99;   // the value was copied
    check("attr: get thread count",
          TAPP_attr_get(attr, TAPP_THREAD_COUNT, &size, &value) == 0 && size == sizeof(int) && *(int*)value == 3);

    cpu_set_t mask, empty;
    CPU_ZERO(&mask); CPU_ZERO(&empty);
    CPU_SET(0, &mask);
    check("attr: set CPU mask", TAPP_attr_set(attr, TAPP_CPU_MASK, sizeof(mask), &mask) == 0);
    check("attr: get CPU mask",
          TAPP_attr_get(attr, TAPP_CPU_MASK, &size, &value) == 0 && size == sizeof(cpu_set_t) && CPU_EQUAL(&mask, (cpu_set_t*)value));

    int zero = 0;
    check("attr: unknown key is refused", TAPP_attr_set(attr, 12345, sizeof(count), &count) != 0);
    check("attr: 0 threads is refused", TAPP_attr_set(attr, TAPP_THREAD_COUNT, sizeof(zero), &zero) != 0);
    check("attr: wrong size of thread count is refused", TAPP_attr_set(attr, TAPP_THREAD_COUNT, 1, &count) != 0);
    check("attr: empty CPU mask is refused", TAPP_attr_set(attr, TAPP_CPU_MASK, sizeof(empty), &empty) != 0);
    check("attr: wrong size of CPU mask is refused", TAPP_attr_set(attr, TAPP_CPU_MASK, sizeof(int), &mask) != 0);
    check("attr: refused values leave the old one",
          TAPP_attr_get(attr, TAPP_THREAD_COUNT, &size, &value) == 0 && value != nullptr && *(int*)value == 3);

    check("attr: clear", TAPP_attr_clear(attr, TAPP_THREAD_COUNT) == 0 &&
                         TAPP_attr_get(attr, TAPP_THREAD_COUNT, &size, &value) == 0 && value == nullptr);

    // A CPU that does not exist: the executor must not be created
    cpu_set_t missing;
    CPU_ZERO(&missing);
    CPU_SET(CPU_SETSIZE - 1, &missing);
    TAPP_executor ex;
    TAPP_attr_set(attr, TAPP_CPU_MASK, sizeof(missing), &missing);
    check("attr: executor on a missing CPU is refused", TAPP_create_executor(&ex, attr) != 0);

    // One CPU that exists (the one this thread runs on) and one that does not: no executor on only a part of its mask
    CPU_SET(sched_getcpu(), &missing);
    TAPP_attr_set(attr, TAPP_CPU_MASK, sizeof(missing), &missing);
    check("attr: executor on a partly missing mask is refused", TAPP_create_executor(&ex, attr) != 0);

    check("attr: destroy", TAPP_destroy_attr(attr) == 0);
}

// The CPUs this process may use. Not the mask of the main thread: OpenMP narrows that one to a single CPU when it binds
// threads (OMP_PROC_BIND), so a helper thread tries every CPU
static std::vector<int> usable_cpus()
{
    std::vector<int> cpus;
    std::thread([&cpus] {
        for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
        {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpu, &one);
            if (sched_setaffinity(0, sizeof(one), &one) == 0) cpus.push_back(cpu);
        }
    }).join();
    return cpus;
}

// Several executors at once, each on two CPUs of its own, with many products queued on each
static void test_executors(const std::vector<int>& cpus)
{
    const int n = 64, tasks = 20, rounds = 5;
    const int executors = std::min<int>(3, cpus.size() / 2);
    if (executors == 0) { std::printf("skip executors: needs two CPUs\n"); return; }

    // A = 1 and B = 2 everywhere, D starts as 1: every element of D becomes alpha * 2 * n + beta
    const int64_t i = 1, j = 2, k = 3;
    auto a = colmajor({n, n}, {i, k}), b = colmajor({n, n}, {k, j}), d = colmajor({n, n}, {i, j});
    std::vector<double> A(n * n, 1.0), B(n * n, 2.0);
    std::vector<std::vector<double>> D(executors * tasks, std::vector<double>(n * n));
    auto is = [](const std::vector<double>& D, double value) {
        return std::all_of(D.begin(), D.end(), [value](double x) { return std::abs(x - value) < 1e-9; });
    };

    std::vector<TAPP_executor> ex(executors);
    std::vector<cpu_set_t> masks(executors);
    std::vector<int> on_mask_before(executors);
    bool created = true;
    wait_for_exited_threads();
    for (int e = 0; e < executors; e++)
    {
        CPU_ZERO(&masks[e]);
        CPU_SET(cpus[2 * e], &masks[e]);
        CPU_SET(cpus[2 * e + 1], &masks[e]);
        on_mask_before[e] = count_threads(&masks[e]);

        TAPP_attr attr;
        TAPP_create_attr(&attr);
        TAPP_attr_set(attr, TAPP_CPU_MASK, sizeof(cpu_set_t), &masks[e]);
        if (TAPP_create_executor(&ex[e], attr) != 0) created = false;
        TAPP_destroy_attr(attr);
    }
    check("executors: create", created);
    if (!created) return;

    // One plan, created with the first executor and executed on all of them
    TAPP_tensor_info ia = info(a, TAPP_F64), ib = info(b, TAPP_F64), id = info(d, TAPP_F64);
    TAPP_tensor_product plan;
    TAPP_create_tensor_product(&plan, ex[0], TAPP_IDENTITY, ia, a.idx.data(), TAPP_IDENTITY, ib, b.idx.data(),
                               TAPP_IDENTITY, id, d.idx.data(), TAPP_IDENTITY, id, d.idx.data(), TAPP_DEFAULT_PREC);
    TAPP_destroy_tensor_info(ia); TAPP_destroy_tensor_info(ib); TAPP_destroy_tensor_info(id);

    bool waited_one = true, waited_all = true;
    std::vector<TAPP_status> st(executors * tasks);
    for (int round = 0; round < rounds; round++)
    {
        for (auto& x : D) std::fill(x.begin(), x.end(), 1.0);
        for (int t = 0; t < tasks; t++)
        {
            for (int e = 0; e < executors; e++)
            {
                double alpha = t + 1, beta = 0.5;   // locals: read when the product is submitted
                double* Dp = D[e * tasks + t].data();
                TAPP_execute_product(plan, ex[e], &st[e * tasks + t], &alpha, A.data(), B.data(), &beta, Dp, Dp);
            }
        }

        // A product in the middle of each queue: it must be done when the wait returns
        for (int e = 0; e < executors; e++)
        {
            int t = tasks / 2;
            TAPP_wait_one(ex[e], st[e * tasks + t]);
            if (!is(D[e * tasks + t], (t + 1) * 2.0 * n + 0.5)) waited_one = false;
        }

        for (int e = 0; e < executors; e++) TAPP_wait_all(ex[e]);
        for (int e = 0; e < executors; e++)
        {
            for (int t = 0; t < tasks; t++)
            {
                if (!is(D[e * tasks + t], (t + 1) * 2.0 * n + 0.5)) waited_all = false;
            }
        }
    }
    check("executors: wait for one product", waited_one);
    check("executors: wait for all products", waited_all);

    // The executors are still alive, so their threads are too: two per executor, on the mask of that executor
    bool on_masks = true;
    for (int e = 0; e < executors; e++)
    {
        if (count_threads(&masks[e]) - on_mask_before[e] != limited(2)) on_masks = false;
    }
    check("executors: every executor runs on its own CPU mask", on_masks);

    // Destroying an executor finishes the products that are still queued
    for (auto& x : D) std::fill(x.begin(), x.end(), 1.0);
    for (int t = 0; t < tasks; t++)
    {
        double alpha = 1.0, beta = 0.0;
        TAPP_execute_product(plan, ex[0], nullptr, &alpha, A.data(), B.data(), &beta, D[t].data(), D[t].data());
    }
    for (int e = 0; e < executors; e++) TAPP_destroy_executor(ex[e]);
    bool drained = true;
    for (int t = 0; t < tasks; t++)
    {
        if (!is(D[t], 2.0 * n)) drained = false;
    }
    check("executors: destroy finishes the queued products", drained);

    TAPP_destroy_tensor_product(plan);
}

// Whether a plan for A * B -> D can be created. A plan that is created is destroyed again
static bool plan_created(const T& a, const T& b, const T& d)
{
    TAPP_tensor_info ia = info(a, TAPP_F64), ib = info(b, TAPP_F64), id = info(d, TAPP_F64);
    TAPP_executor ex;
    TAPP_create_executor(&ex, 0);
    TAPP_tensor_product plan;
    TAPP_error e = TAPP_create_tensor_product(&plan, ex, TAPP_IDENTITY, ia, a.idx.data(), TAPP_IDENTITY, ib, b.idx.data(),
                                              TAPP_IDENTITY, id, d.idx.data(), TAPP_IDENTITY, id, d.idx.data(),
                                              TAPP_DEFAULT_PREC);
    TAPP_destroy_executor(ex);
    TAPP_destroy_tensor_info(ia); TAPP_destroy_tensor_info(ib); TAPP_destroy_tensor_info(id);
    if (e == 0) TAPP_destroy_tensor_product(plan);
    return e == 0;
}

// Plans whose indices do not describe a product are refused
static void test_plan_indices()
{
    const int64_t i = 1, j = 2, k = 3, m = 4;
    auto a = colmajor({5, 4}, {i, k}), b = colmajor({4, 3}, {k, j}), d = colmajor({5, 3}, {i, j});

    check("plan: a contraction is accepted", plan_created(a, b, d));
    check("plan: an index repeated in A with one extent is accepted",
          plan_created(colmajor({4, 4}, {k, k}), b, colmajor({3}, {j})));

    check("plan: different extents of an index in A and B are refused", !plan_created(a, colmajor({3, 3}, {k, j}), d));
    check("plan: different extents of an index in A and D are refused", !plan_created(a, b, colmajor({6, 3}, {i, j})));
    check("plan: different extents of an index within A are refused",
          !plan_created(colmajor({5, 4}, {k, k}), b, colmajor({3}, {j})));
    check("plan: an index of D that is not in A or B is refused", !plan_created(a, b, colmajor({5, 3, 2}, {i, j, m})));
}

// Whether a permute plan for A, B -> C can be created. A plan that is created is destroyed again
static bool permute_created(const T& a, const T& b, const T& c, TAPP_datatype type_A = TAPP_F64)
{
    TAPP_tensor_info ia = info(a, type_A), ib = info(b, TAPP_F64), ic = info(c, TAPP_F64);
    TAPP_tensor_permute plan;
    TAPP_error e = TAPP_create_tensor_permute(&plan, TAPP_IDENTITY, ia, a.idx.data(), TAPP_IDENTITY, ib, b.idx.data(),
                                              TAPP_IDENTITY, ic, c.idx.data(), TAPP_DEFAULT_PREC);
    TAPP_destroy_tensor_info(ia); TAPP_destroy_tensor_info(ib); TAPP_destroy_tensor_info(ic);
    if (e == 0) TAPP_destroy_tensor_permute(plan);
    return e == 0;
}

// A permute only reorders: A and C have the same indices, each once, with the same extents, and B has the structure of C
static void test_permute_plan()
{
    const int64_t i = 1, j = 2, k = 3;
    auto a = colmajor({5, 4}, {i, j}), c = colmajor({4, 5}, {j, i});

    check("permute plan: a transpose is accepted", permute_created(a, c, c));
    check("permute plan: the same index order is accepted", permute_created(a, a, a));
    check("permute plan: scalars are accepted", permute_created(colmajor({}, {}), colmajor({}, {}), colmajor({}, {})));

    check("permute plan: an index of A that is not in C is refused",
          !permute_created(a, colmajor({5}, {i}), colmajor({5}, {i})));
    check("permute plan: an index of C that is not in A is refused",
          !permute_created(a, colmajor({4, 5, 2}, {j, i, k}), colmajor({4, 5, 2}, {j, i, k})));
    check("permute plan: an index repeated in A is refused",
          !permute_created(colmajor({4, 4}, {i, i}), colmajor({4}, {i}), colmajor({4}, {i})));
    check("permute plan: an index repeated in C is refused",
          !permute_created(a, colmajor({5, 5}, {i, i}), colmajor({5, 5}, {i, i})));
    check("permute plan: different extents of an index in A and C are refused",
          !permute_created(a, colmajor({5, 5}, {j, i}), colmajor({5, 5}, {j, i})));
    check("permute plan: B with other strides than C is refused", !permute_created(a, rowmajor({4, 5}, {j, i}), c));
    check("permute plan: B with another index order than C is refused", !permute_created(a, colmajor({5, 4}, {i, j}), c));
    check("permute plan: different datatypes are refused", !permute_created(a, c, c, TAPP_F32));
}

// Pointers to the data of A and C are required, B only when beta != 0
static void test_permute_null()
{
    const int64_t i = 1, j = 2;
    auto a = colmajor({5, 4}, {i, j}), c = colmajor({4, 5}, {j, i});
    std::vector<double> A(20, 1.0), B(20, 1.0), C(20, 0.0);
    double alpha = 1, beta = 1, zero = 0;

    TAPP_executor ex;
    TAPP_create_executor(&ex, 0);
    TAPP_tensor_info ia = info(a, TAPP_F64), ic = info(c, TAPP_F64);
    TAPP_tensor_permute plan;
    TAPP_create_tensor_permute(&plan, TAPP_IDENTITY, ia, a.idx.data(), TAPP_IDENTITY, ic, c.idx.data(),
                               TAPP_IDENTITY, ic, c.idx.data(), TAPP_DEFAULT_PREC);
    TAPP_destroy_tensor_info(ia); TAPP_destroy_tensor_info(ic);

    check("permute: B = NULL with beta != 0 is refused",
          TAPP_execute_permute(plan, ex, nullptr, &alpha, A.data(), &beta, nullptr, C.data()) != 0);
    check("permute: B = NULL with beta = 0 is accepted",
          TAPP_execute_permute(plan, ex, nullptr, &alpha, A.data(), &zero, nullptr, C.data()) == 0);
    check("permute: A = NULL is refused",
          TAPP_execute_permute(plan, ex, nullptr, &alpha, nullptr, &beta, B.data(), C.data()) != 0);
    check("permute: C = NULL is refused",
          TAPP_execute_permute(plan, ex, nullptr, &alpha, A.data(), &beta, B.data(), nullptr) != 0);
    TAPP_wait_all(ex);

    TAPP_destroy_tensor_permute(plan);
    TAPP_destroy_executor(ex);
}

// A product does not need its plan after it is submitted: every plan is destroyed while its product is still queued
static void test_plan_lifetime()
{
    const int n = 64, tasks = 20;
    const int64_t i = 1, j = 2, k = 3;
    auto a = colmajor({n, n}, {i, k}), b = colmajor({n, n}, {k, j}), d = colmajor({n, n}, {i, j});
    std::vector<double> A(n * n, 1.0), B(n * n, 2.0);
    std::vector<std::vector<double>> D(tasks, std::vector<double>(n * n, 1.0));

    TAPP_executor ex;
    TAPP_create_executor(&ex, 0);
    for (int t = 0; t < tasks; t++)
    {
        TAPP_tensor_info ia = info(a, TAPP_F64), ib = info(b, TAPP_F64), id = info(d, TAPP_F64);
        TAPP_tensor_product plan;
        TAPP_create_tensor_product(&plan, ex, TAPP_IDENTITY, ia, a.idx.data(), TAPP_IDENTITY, ib, b.idx.data(),
                                   TAPP_IDENTITY, id, d.idx.data(), TAPP_IDENTITY, id, d.idx.data(), TAPP_DEFAULT_PREC);
        TAPP_destroy_tensor_info(ia); TAPP_destroy_tensor_info(ib); TAPP_destroy_tensor_info(id);

        double alpha = t + 1, beta = 0.5;
        TAPP_execute_product(plan, ex, nullptr, &alpha, A.data(), B.data(), &beta, D[t].data(), D[t].data());
        TAPP_destroy_tensor_product(plan);
    }
    TAPP_wait_all(ex);

    // A = 1 and B = 2 everywhere, D started as 1: every element of D is alpha * 2 * n + beta
    bool ok = true;
    for (int t = 0; t < tasks; t++)
    {
        for (double x : D[t]) if (std::abs(x - ((t + 1) * 2.0 * n + 0.5)) > 1e-9) ok = false;
    }
    check("plan: destroyed before its product has run", ok);

    // The executor has run the products 1 to tasks: any other status is not one of its products
    check("status: the last product can be waited for", TAPP_wait_one(ex, tasks) == 0);
    check("status: 0 is refused", TAPP_wait_one(ex, 0) != 0);
    check("status: a negative one is refused", TAPP_wait_one(ex, -1) != 0);
    check("status: one that was never given out is refused", TAPP_wait_one(ex, tasks + 1) != 0);

    TAPP_destroy_executor(ex);
}

// The executor is still alive, so its threads are too: the worker and the ones it started for the operation
static void check_executor_threads(int threads_before, int on_mask_before)
{
    if (exec_threads >= 0)
    {
        int started = count_threads(nullptr) - threads_before;
        std::printf("%s   %d thread(s), expected %d\n", started == exec_threads ? "ok  " : "FAIL", started, exec_threads);
        if (started != exec_threads) failures++;
    }
    if (exec_threads >= 0 && exec_mask)
    {
        int on_mask = count_threads(exec_mask) - on_mask_before;
        std::printf("%s   %d thread(s) on the CPU mask, expected %d\n", on_mask == exec_threads ? "ok  " : "FAIL", on_mask, exec_threads);
        if (on_mask != exec_threads) failures++;
    }
}

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

    if (exec_threads >= 0) wait_for_exited_threads();
    int threads_before = count_threads(nullptr), on_mask_before = exec_mask ? count_threads(exec_mask) : 0;
    TAPP_executor ex;
    TAPP_error ee = TAPP_create_executor(&ex, exec_attr);
    if (ee != 0) { std::printf("FAIL %-45s create_executor returned %d\n", name, ee); failures++; return; }
    TAPP_tensor_info ia = info(a, dtype<V>()), ib = info(b, dtype<V>()), ic = info(d, dtype<V>()), id = info(d, dtype<V>());
    TAPP_tensor_product plan;
    TAPP_error e = TAPP_create_tensor_product(&plan, ex, opA, ia, a.idx.data(), opB, ib, b.idx.data(),
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

    check_executor_threads(threads_before, on_mask_before);

    TAPP_destroy_tensor_product(plan);
    TAPP_destroy_executor(ex);
}

// in_place: pass C as B. b_null: pass NULL for B (beta must be 0). nan_C: prefill C with NaN (not with padding: the
// padding would stay NaN).
// tblis_88: a case that TBLIS 9b95712 computes wrong (TBLIS issue #88: a conjugated A, beta != 0 and a change of index
// order). A wrong result is reported but not counted as a failure, a right one says that the issue seems to be fixed.
template <typename V>
static void run_permute(const char* name, const T& a, const T& c, int opA, int opB, int opC, V alpha, V beta,
                        bool in_place = false, bool b_null = false, bool nan_C = false, bool tblis_88 = false)
{
    std::mt19937 g(42);
    std::vector<V> A(a.span()), B(c.span()), C(c.span());
    for (auto& x : A) x = rnd<V>(g);
    for (auto& x : B) x = rnd<V>(g);
    for (auto& x : C) x = nan_C ? V(std::numeric_limits<double>::quiet_NaN()) : rnd<V>(g);
    if (in_place) C = B;

    // Elements in the padding are compared too: they must keep their value
    std::vector<V> R = C;
    reference_permute(a, A, opA, (b_null ? nullptr : &B), opB, c, R, opC, alpha, beta);

    if (exec_threads >= 0) wait_for_exited_threads();
    int threads_before = count_threads(nullptr), on_mask_before = exec_mask ? count_threads(exec_mask) : 0;
    TAPP_executor ex;
    TAPP_error ee = TAPP_create_executor(&ex, exec_attr);
    if (ee != 0) { std::printf("FAIL %-45s create_executor returned %d\n", name, ee); failures++; return; }
    TAPP_tensor_info ia = info(a, dtype<V>()), ib = info(c, dtype<V>()), ic = info(c, dtype<V>());
    TAPP_tensor_permute plan;
    TAPP_error e = TAPP_create_tensor_permute(&plan, opA, ia, a.idx.data(), opB, ib, c.idx.data(),
                                              opC, ic, c.idx.data(), TAPP_DEFAULT_PREC);
    if (e != 0) { std::printf("FAIL %-45s create_tensor_permute returned %d\n", name, e); failures++; return; }

    // Destroy the user's tensor infos right away: the plan must not depend on them
    TAPP_destroy_tensor_info(ia); TAPP_destroy_tensor_info(ib); TAPP_destroy_tensor_info(ic);

    TAPP_status st = 0;
    {
        V alpha_local = alpha, beta_local = beta;   // goes out of scope before the wait, like a caller's locals would
        const void* Bp = b_null ? nullptr : (in_place ? (const void*)C.data() : (const void*)B.data());
        e = TAPP_execute_permute(plan, ex, &st, &alpha_local, A.data(), &beta_local, Bp, C.data());
        alpha_local = V(999); beta_local = V(999);
    }
    if (e != 0) { std::printf("FAIL %-45s execute_permute returned %d\n", name, e); failures++; return; }
    TAPP_wait_one(ex, st);

    double err = 0, scale = 0;
    bool nan = false;
    for (size_t i = 0; i < C.size(); i++)
    {
        if (std::isnan(std::abs(C[i]))) nan = true;
        err = std::max(err, (double)std::abs(C[i] - R[i]));
        scale = std::max(scale, (double)std::abs(R[i]));
    }
    double tol = (sizeof(V) == sizeof(float) ? 1e-5 : 1e-12) * std::max(1.0, scale);
    bool ok = !nan && err <= tol;
    if (tblis_88)
    {
        std::printf("%s %-45s max error %.2e%s\n", ok ? "ok  " : "known", name, err,
                    ok ? " (TBLIS #88 seems fixed: remove the known issue)" : " (TBLIS #88)");
    }
    else
    {
        std::printf("%s %-45s max error %.2e%s\n", ok ? "ok  " : "FAIL", name, err, nan ? " (NaN in C)" : "");
        if (!ok) failures++;
    }

    check_executor_threads(threads_before, on_mask_before);

    TAPP_destroy_tensor_permute(plan);
    TAPP_destroy_executor(ex);
}

int main()
{
    TAPP_start();

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

    // D transposed: B and D have different strides for j, where TBLIS conjugates a conjugated output wrong and writes
    // outside of it (TBLIS #91). D must never reach tblis_tensor_mult conjugated
    auto dt = colmajor({3, 5}, {j, i});
    run<cd>("c64 gemm, D transposed, conj C", a, b, dt, I, I, CJ, I, alpha, beta);
    run<cd>("c64 gemm, D transposed, conj C, in place", a, b, dt, I, I, CJ, I, alpha, beta, true);

    // C and D with unused elements after every column: C must be copied into D along their strides
    auto dp = padded({5, 3}, {i, j}, 2);
    run<double>("f64 gemm, padded C and D", a, b, dp, I, I, I, I, 1.5, 0.5);
    run<double>("f64 gemm, padded C and D, in place", a, b, dp, I, I, I, I, 1.5, 0.5, true);
    run<cd>("c64 gemm, padded C and D, conj C", a, b, dp, I, I, CJ, I, alpha, beta);

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

    // Permutes: C = op_C(alpha * op_A(A) + beta * op_B(B)), where B has the structure of C
    auto pa = colmajor({5, 4}, {i, j}), pt = colmajor({4, 5}, {j, i}), pp = padded({4, 5}, {j, i}, 3);
    run_permute<double>("f64 permute, transpose", pa, pt, I, I, I, 1.5, 0.5);
    run_permute<float>("f32 permute, transpose", pa, pt, I, I, I, 1.5f, 0.5f);
    run_permute<double>("f64 permute, same index order", pa, pa, I, I, I, 1.5, 0.5);
    run_permute<double>("f64 permute, ijk -> kij", colmajor({3, 4, 5}, {i, j, k}), colmajor({5, 3, 4}, {k, i, j}),
                        I, I, I, 1.5, 0.5);
    run_permute<double>("f64 permute, scalar", colmajor({}, {}), colmajor({}, {}), I, I, I, 1.5, 0.5);
    run_permute<double>("f64 permute, in place (B == C)", pa, pt, I, I, I, 1.5, 0.5, true);
    run_permute<double>("f64 permute, beta = 0, B = NULL", pa, pt, I, I, I, 2.0, 0.0, false, true);
    run_permute<double>("f64 permute, beta = 0, C prefilled with NaN", pa, pt, I, I, I, 2.0, 0.0, false, true, true);
    run_permute<double>("f64 permute, padded B and C", pa, pp, I, I, I, 1.5, 0.5);

    // Every combination of conjugations, with the same index order and transposed. TBLIS conjugates A when op_A and
    // op_C differ, which it gets wrong in a transpose with beta != 0 (TBLIS #88)
    for (bool transpose : {false, true})
    for (int op = 0; op < 8; op++)
    {
        int opA = op & 1 ? CJ : I, opB = op & 2 ? CJ : I, opC = op & 4 ? CJ : I;
        char name[64];
        std::snprintf(name, sizeof(name), "c64 permute, %s, conj%s%s%s%s", transpose ? "transpose" : "same order",
                      opA ? " A" : "", opB ? " B" : "", opC ? " C" : "", op ? "" : " none");
        run_permute<cd>(name, pa, transpose ? pt : pa, opA, opB, opC, alpha, beta, false, false, false,
                        transpose && opA != opC);
    }
    run_permute<cd>("c64 permute, transpose, conj A, beta = 0", pa, pt, CJ, I, I, alpha, cd(0, 0), false, true);
    run_permute<cd>("c64 permute, same order, conj B, in place", pa, pa, I, CJ, I, alpha, beta, true);
    run_permute<cd>("c64 permute, transpose, conj B, in place", pa, pt, I, CJ, I, alpha, beta, true);
    run_permute<cd>("c64 permute, transpose, conj B, padded", pa, pp, I, CJ, I, alpha, beta);

    test_attributes();

    // Executors with their own resources, on tensors large enough for TBLIS to split the work between threads
    auto al = colmajor({30, 20, 16}, {i, k, l}), bl = colmajor({20, 16, 30}, {k, l, j}), dl = colmajor({30, 30}, {i, j});
    auto pl = colmajor({30, 20, 16}, {i, k, l}), plt = colmajor({16, 30, 20}, {l, i, k});

    // The mask holds the first two CPUs this process may use, or the only one
    std::vector<int> cpus = usable_cpus();
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (size_t c = 0; c < cpus.size() && c < 2; c++) CPU_SET(cpus[c], &mask);
    int count = 3;

    TAPP_create_attr(&exec_attr);

    TAPP_attr_set(exec_attr, TAPP_THREAD_COUNT, sizeof(count), &count);
    exec_threads = limited(count);
    run<double>("f64 contraction, thread count", al, bl, dl, I, I, I, I, 1.5, 0.5);
    run<cd>("c64 contraction, conj C, thread count", al, bl, dl, I, I, CJ, I, alpha, beta);
    run_permute<double>("f64 permute, thread count", pl, plt, I, I, I, 1.5, 0.5);

    TAPP_attr_set(exec_attr, TAPP_CPU_MASK, sizeof(mask), &mask);
    exec_mask = &mask;
    run<double>("f64 contraction, thread count and CPU mask", al, bl, dl, I, I, I, I, 1.5, 0.5);
    run<cd>("c64 contraction, conj all, count and mask", al, bl, dl, CJ, CJ, CJ, CJ, alpha, beta);
    run_permute<cd>("c64 permute, conj B, count and mask", pl, plt, I, CJ, I, alpha, beta);

    TAPP_attr_clear(exec_attr, TAPP_THREAD_COUNT);
    exec_threads = limited(CPU_COUNT(&mask));
    run<double>("f64 contraction, CPU mask", al, bl, dl, I, I, I, I, 1.5, 0.5);
    run<cd>("c64 contraction, conj C, CPU mask", al, bl, dl, I, I, CJ, I, alpha, beta);

    TAPP_destroy_attr(exec_attr);
    exec_attr = 0; exec_threads = -1; exec_mask = nullptr;

    test_executors(cpus);
    test_plan_lifetime();
    test_plan_indices();
    test_permute_plan();
    test_permute_null();

    TAPP_shutdown();

    std::printf("\n%d failure(s)\n", failures);
    return failures != 0;
}
