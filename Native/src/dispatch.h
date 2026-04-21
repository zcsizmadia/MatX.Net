// dispatch.h  –  Compile-time type + rank dispatch helpers.
// Generates all (dtype × rank) combinations via X-macros.
#pragma once

#include "tensor_handle.h"
#include "executor_handle.h"

// ── X-macro lists ─────────────────────────────────────────────────────────────

/// All supported dtypes.
#define MATX_FOREACH_DTYPE(X) \
    X(float,    MATX_DTYPE_FLOAT32)   \
    X(double,   MATX_DTYPE_FLOAT64)   \
    X(cf32_t,   MATX_DTYPE_COMPLEX64) \
    X(cd64_t,   MATX_DTYPE_COMPLEX128)\
    X(int32_t,  MATX_DTYPE_INT32)     \
    X(int64_t,  MATX_DTYPE_INT64)

/// Floating-point + complex dtypes only.
#define MATX_FOREACH_FLOAT_DTYPE(X) \
    X(float,    MATX_DTYPE_FLOAT32)   \
    X(double,   MATX_DTYPE_FLOAT64)   \
    X(cf32_t,   MATX_DTYPE_COMPLEX64) \
    X(cd64_t,   MATX_DTYPE_COMPLEX128)

/// Real-valued floating-point dtypes only.
#define MATX_FOREACH_REAL_FLOAT_DTYPE(X) \
    X(float,    MATX_DTYPE_FLOAT32)      \
    X(double,   MATX_DTYPE_FLOAT64)

/// All supported ranks.
#define MATX_FOREACH_RANK(X) X(1) X(2) X(3) X(4)

// ── Single-tensor visitor ─────────────────────────────────────────────────────
//
//  Usage:
//    visit_tensor(tb, [](auto& t) { ... });
//
//  'Visitor' must accept any matx::tensor_t<T, RANK>& and return void.
//
template<typename Visitor>
void visit_tensor(MatxTensorBase* tb, Visitor&& v) {
#define DISPATCH_DTYPE(T, DENUM)                                           \
    case DENUM:                                                             \
        switch (tb->rank) {                                                 \
            case 1: v(static_cast<MatxTypedTensor<T,1>*>(tb)->t); return;  \
            case 2: v(static_cast<MatxTypedTensor<T,2>*>(tb)->t); return;  \
            case 3: v(static_cast<MatxTypedTensor<T,3>*>(tb)->t); return;  \
            case 4: v(static_cast<MatxTypedTensor<T,4>*>(tb)->t); return;  \
            default: throw std::runtime_error("rank > 4 not supported");   \
        }
    switch (tb->dtype) {
        MATX_FOREACH_DTYPE(DISPATCH_DTYPE)
        default: throw std::runtime_error("unsupported dtype");
    }
#undef DISPATCH_DTYPE
}

/// Variant that iterates only over float/complex dtypes.
template<typename Visitor>
void visit_float_tensor(MatxTensorBase* tb, Visitor&& v) {
#define DISPATCH_DTYPE(T, DENUM)                                           \
    case DENUM:                                                             \
        switch (tb->rank) {                                                 \
            case 1: v(static_cast<MatxTypedTensor<T,1>*>(tb)->t); return;  \
            case 2: v(static_cast<MatxTypedTensor<T,2>*>(tb)->t); return;  \
            case 3: v(static_cast<MatxTypedTensor<T,3>*>(tb)->t); return;  \
            case 4: v(static_cast<MatxTypedTensor<T,4>*>(tb)->t); return;  \
            default: throw std::runtime_error("rank > 4 not supported");   \
        }
    switch (tb->dtype) {
        MATX_FOREACH_FLOAT_DTYPE(DISPATCH_DTYPE)
        default: throw std::runtime_error("dtype must be float or complex");
    }
#undef DISPATCH_DTYPE
}

// ── Two-tensor visitor (same dtype + rank required) ───────────────────────────
//
//  Visitor: void(auto& dst_t, auto& src_t)
//
template<typename Visitor>
void visit_tensor2(MatxTensorBase* a, MatxTensorBase* b, Visitor&& v) {
    if (a->dtype != b->dtype)
        throw std::runtime_error("tensor dtype mismatch");
    if (a->rank != b->rank)
        throw std::runtime_error("tensor rank mismatch");
#define DISPATCH_DTYPE(T, DENUM)                                                                  \
    case DENUM:                                                                                   \
        switch (a->rank) {                                                                        \
            case 1: v(static_cast<MatxTypedTensor<T,1>*>(a)->t,                                  \
                       static_cast<MatxTypedTensor<T,1>*>(b)->t); return;                        \
            case 2: v(static_cast<MatxTypedTensor<T,2>*>(a)->t,                                  \
                       static_cast<MatxTypedTensor<T,2>*>(b)->t); return;                        \
            case 3: v(static_cast<MatxTypedTensor<T,3>*>(a)->t,                                  \
                       static_cast<MatxTypedTensor<T,3>*>(b)->t); return;                        \
            case 4: v(static_cast<MatxTypedTensor<T,4>*>(a)->t,                                  \
                       static_cast<MatxTypedTensor<T,4>*>(b)->t); return;                        \
            default: throw std::runtime_error("rank > 4 not supported");                          \
        }
    switch (a->dtype) {
        MATX_FOREACH_DTYPE(DISPATCH_DTYPE)
        default: throw std::runtime_error("unsupported dtype");
    }
#undef DISPATCH_DTYPE
}

// ── Three-tensor visitor (all same dtype + rank required) ─────────────────────
//
//  Visitor: void(auto& dst_t, auto& a_t, auto& b_t)
//
template<typename Visitor>
void visit_tensor3(MatxTensorBase* dst, MatxTensorBase* a, MatxTensorBase* b, Visitor&& v) {
    if (dst->dtype != a->dtype || a->dtype != b->dtype)
        throw std::runtime_error("tensor dtype mismatch");
    if (dst->rank != a->rank || a->rank != b->rank)
        throw std::runtime_error("tensor rank mismatch");
#define DISPATCH_DTYPE(T, DENUM)                                                                  \
    case DENUM:                                                                                   \
        switch (dst->rank) {                                                                      \
            case 1: v(static_cast<MatxTypedTensor<T,1>*>(dst)->t,                                \
                       static_cast<MatxTypedTensor<T,1>*>(a)->t,                                 \
                       static_cast<MatxTypedTensor<T,1>*>(b)->t); return;                        \
            case 2: v(static_cast<MatxTypedTensor<T,2>*>(dst)->t,                                \
                       static_cast<MatxTypedTensor<T,2>*>(a)->t,                                 \
                       static_cast<MatxTypedTensor<T,2>*>(b)->t); return;                        \
            case 3: v(static_cast<MatxTypedTensor<T,3>*>(dst)->t,                                \
                       static_cast<MatxTypedTensor<T,3>*>(a)->t,                                 \
                       static_cast<MatxTypedTensor<T,3>*>(b)->t); return;                        \
            case 4: v(static_cast<MatxTypedTensor<T,4>*>(dst)->t,                                \
                       static_cast<MatxTypedTensor<T,4>*>(a)->t,                                 \
                       static_cast<MatxTypedTensor<T,4>*>(b)->t); return;                        \
            default: throw std::runtime_error("rank > 4 not supported");                          \
        }
    switch (dst->dtype) {
        MATX_FOREACH_DTYPE(DISPATCH_DTYPE)
        default: throw std::runtime_error("unsupported dtype");
    }
#undef DISPATCH_DTYPE
}

// ── Error boundary macros ─────────────────────────────────────────────────────
#define MATX_TRY  try {
#define MATX_CATCH(ret)                                          \
    } catch (const std::exception& _e) {                         \
        matx_set_last_error(_e.what());                          \
        return (ret);                                            \
    } catch (...) {                                              \
        matx_set_last_error("unknown error");                    \
        return (ret);                                            \
    }

#define MATX_CATCH_HANDLE(ret)                                   \
    } catch (const std::exception& _e) {                         \
        matx_set_last_error(_e.what());                          \
        return (ret);                                            \
    } catch (...) {                                              \
        matx_set_last_error("unknown error");                    \
        return (ret);                                            \
    }

// Declared in matx_core.cu, implemented once.
extern "C" void matx_set_last_error(const char* msg);

// ── Cast helpers ─────────────────────────────────────────────────────────────
inline MatxTensorBase* to_base(MatxTensorHandle h) {
    return static_cast<MatxTensorBase*>(h);
}
inline ExecutorHandle* to_exec(MatxExecutorHandle h) {
    return static_cast<ExecutorHandle*>(h);
}
