// matx_elementwise.cu  –  Element-wise unary and binary operations.
#include "dispatch.h"
#include <cuda/std/cmath>

// ── Back-fill MatX operators removed from upstream ───────────────────────────
// erf, erfc, and sign were removed from matx namespace; define them locally
// using the same MATX_UNARY_OP_GEN_NOFUNC / MATX_DEFINE_UNARY_OP pattern.
namespace matx { namespace detail {

template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_erf(T v1) {
    if constexpr (cuda::std::is_floating_point_v<T>) { return cuda::std::erf(v1); }
    else { return v1; }
}
template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_erfc(T v1) {
    if constexpr (cuda::std::is_floating_point_v<T>) { return cuda::std::erfc(v1); }
    else { return v1; }
}
template <typename T>
static __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto scalar_internal_sign(T v1) {
    return (v1 > T(0)) ? T(1) : ((v1 < T(0)) ? T(-1) : T(0));
}

MATX_UNARY_OP_GEN_NOFUNC(erf,  Erf)
MATX_UNARY_OP_GEN_NOFUNC(erfc, Erfc)
MATX_UNARY_OP_GEN_NOFUNC(sign, Sign)

} } // namespace matx::detail

namespace matx {
    MATX_DEFINE_UNARY_OP(erf,  detail::ErfOp)
    MATX_DEFINE_UNARY_OP(erfc, detail::ErfcOp)
    MATX_DEFINE_UNARY_OP(sign, detail::SignOp)
} // namespace matx

// ── Unary helper macro ────────────────────────────────────────────────────────
//
//  UNARY_OP(name, matx_expr)
//    name      – C API function suffix, e.g. "abs"
//    matx_expr – MatX expression applied to the tensor, e.g. matx::abs(a)
//
#define IMPL_UNARY(fn_name, matx_expr)                                         \
extern "C" int fn_name(MatxExecutorHandle exec,                                \
                        MatxTensorHandle dst, MatxTensorHandle src) {          \
    MATX_TRY                                                                   \
        auto* d = to_base(dst);                                                \
        auto* s = to_base(src);                                                \
        visit_tensor(d, [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {     \
            visit_tensor(s, [&]<typename U, int S>(matx::tensor_t<U,S>& st) { \
                if constexpr (std::is_same_v<T,U> && R == S) {                \
                    to_exec(exec)->run([&](auto& ex) {                         \
                        (dt = matx_expr).run(ex);                              \
                    });                                                         \
                } else {                                                       \
                    throw std::runtime_error("dtype/rank mismatch");           \
                }                                                              \
            });                                                                \
        });                                                                    \
        return MATX_OK;                                                        \
    MATX_CATCH(MATX_ERR_TYPE)                                                  \
}

IMPL_UNARY(matx_abs,     matx::abs(st))
IMPL_UNARY(matx_abs2,    matx::abs2(st))
IMPL_UNARY(matx_sqrt,    matx::sqrt(st))
IMPL_UNARY(matx_exp,     matx::exp(st))
IMPL_UNARY(matx_log,     matx::log(st))
IMPL_UNARY(matx_log2,    matx::log2(st))
IMPL_UNARY(matx_log10,   matx::log10(st))
IMPL_UNARY(matx_sin,     matx::sin(st))
IMPL_UNARY(matx_cos,     matx::cos(st))
IMPL_UNARY(matx_tan,     matx::tan(st))
IMPL_UNARY(matx_asin,    matx::asin(st))
IMPL_UNARY(matx_acos,    matx::acos(st))
IMPL_UNARY(matx_atan,    matx::atan(st))
IMPL_UNARY(matx_sinh,    matx::sinh(st))
IMPL_UNARY(matx_cosh,    matx::cosh(st))
IMPL_UNARY(matx_tanh,    matx::tanh(st))
IMPL_UNARY(matx_ceil,    matx::ceil(st))
IMPL_UNARY(matx_floor,   matx::floor(st))
IMPL_UNARY(matx_round,   matx::round(st))
IMPL_UNARY(matx_sign,    matx::sign(st))
IMPL_UNARY(matx_neg,     -st)
IMPL_UNARY(matx_conj,    matx::conj(st))
IMPL_UNARY(matx_real,    matx::real(st))
IMPL_UNARY(matx_imag,    matx::imag(st))
IMPL_UNARY(matx_normcdf, matx::normcdf(st))
IMPL_UNARY(matx_erf,     matx::erf(st))
IMPL_UNARY(matx_erfc,    matx::erfc(st))

#undef IMPL_UNARY

// ── Binary tensor–tensor helper ───────────────────────────────────────────────
#define IMPL_BINARY(fn_name, op_expr)                                          \
extern "C" int fn_name(MatxExecutorHandle exec,                                \
                        MatxTensorHandle dst,                                  \
                        MatxTensorHandle a, MatxTensorHandle b) {              \
    MATX_TRY                                                                   \
        visit_tensor3(to_base(dst), to_base(a), to_base(b),                   \
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt,                   \
                                   matx::tensor_t<T,R>& at,                   \
                                   matx::tensor_t<T,R>& bt) {                 \
                to_exec(exec)->run([&](auto& ex) {                             \
                    (dt = op_expr).run(ex);                                    \
                });                                                             \
            });                                                                \
        return MATX_OK;                                                        \
    MATX_CATCH(MATX_ERR_TYPE)                                                  \
}

IMPL_BINARY(matx_add,     at + bt)
IMPL_BINARY(matx_sub,     at - bt)
IMPL_BINARY(matx_mul,     at * bt)
IMPL_BINARY(matx_div,     at / bt)
IMPL_BINARY(matx_pow,     matx::pow(at, bt))
IMPL_BINARY(matx_maximum, matx::max(at, bt))
IMPL_BINARY(matx_minimum, matx::min(at, bt))

#undef IMPL_BINARY

// ── Scalar variants ───────────────────────────────────────────────────────────
#define IMPL_BINARY_SCALAR(fn_name, op_expr)                                   \
extern "C" int fn_name(MatxExecutorHandle exec,                                \
                        MatxTensorHandle dst, MatxTensorHandle a, double s) {  \
    MATX_TRY                                                                   \
        visit_tensor2(to_base(dst), to_base(a),                               \
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt,                   \
                                   matx::tensor_t<T,R>& at) {                 \
                T scalar = static_cast<T>(s);                                  \
                to_exec(exec)->run([&](auto& ex) {                             \
                    (dt = op_expr).run(ex);                                    \
                });                                                             \
            });                                                                \
        return MATX_OK;                                                        \
    MATX_CATCH(MATX_ERR_TYPE)                                                  \
}

IMPL_BINARY_SCALAR(matx_add_scalar, at + scalar)
IMPL_BINARY_SCALAR(matx_sub_scalar, at - scalar)
IMPL_BINARY_SCALAR(matx_mul_scalar, at * scalar)
IMPL_BINARY_SCALAR(matx_div_scalar, at / scalar)
IMPL_BINARY_SCALAR(matx_pow_scalar, matx::pow(at, scalar))

#undef IMPL_BINARY_SCALAR

// ── Transpose / hermitian ─────────────────────────────────────────────────────
extern "C" int matx_transpose(MatxExecutorHandle exec,
                               MatxTensorHandle dst, MatxTensorHandle src) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(src),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt,
                                   matx::tensor_t<T,R>& st) {
                if constexpr (R == 2) {
                    to_exec(exec)->run([&](auto& ex) {
                        (dt = matx::transpose(st)).run(ex);
                    });
                } else {
                    throw std::runtime_error("transpose requires rank-2 tensors");
                }
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_hermitian(MatxExecutorHandle exec,
                               MatxTensorHandle dst, MatxTensorHandle src) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(src),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt,
                                   matx::tensor_t<T,R>& st) {
                if constexpr (R == 2) {
                    to_exec(exec)->run([&](auto& ex) {
                        (dt = matx::hermitianT(st)).run(ex);
                    });
                } else {
                    throw std::runtime_error("hermitian requires rank-2 tensors");
                }
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Reductions ────────────────────────────────────────────────────────────────

extern "C" int matx_sum(MatxExecutorHandle exec,
                         MatxTensorHandle dst, MatxTensorHandle src, int axis) {
    MATX_TRY
        visit_float_tensor(to_base(src), [&]<typename T, int R>(matx::tensor_t<T,R>& st) {
            visit_float_tensor(to_base(dst), [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) {
                if constexpr (!std::is_same_v<T, DT>)
                    throw std::runtime_error("dst/src dtype mismatch");
                to_exec(exec)->run([&](auto& ex) {
                    if (axis == -1) {
                        (dt = matx::sum(st)).run(ex);
                    } else {
                        (dt = matx::sum(st, {axis})).run(ex);
                    }
                });
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

#define IMPL_REDUCTION(fn_name, matx_fn)                                       \
extern "C" int fn_name(MatxExecutorHandle exec,                                \
                        MatxTensorHandle dst, MatxTensorHandle src, int axis) {\
    MATX_TRY                                                                   \
        visit_float_tensor(to_base(src), [&]<typename T, int R>(matx::tensor_t<T,R>& st) { \
            visit_float_tensor(to_base(dst), [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) { \
                if constexpr (!std::is_same_v<T, DT>)                         \
                    throw std::runtime_error("dst/src dtype mismatch");        \
                to_exec(exec)->run([&](auto& ex) {                             \
                    if (axis == -1) (dt = matx::matx_fn(st)).run(ex);         \
                    else            (dt = matx::matx_fn(st, {axis})).run(ex); \
                });                                                             \
            });                                                                \
        });                                                                    \
        return MATX_OK;                                                        \
    MATX_CATCH(MATX_ERR_TYPE)                                                  \
}

IMPL_REDUCTION(matx_mean, mean)
IMPL_REDUCTION(matx_max_r, max)
IMPL_REDUCTION(matx_min_r, min)
IMPL_REDUCTION(matx_prod, prod)
IMPL_REDUCTION(matx_var,  var)
IMPL_REDUCTION(matx_stdd, stdd)

#undef IMPL_REDUCTION

extern "C" int matx_any(MatxExecutorHandle exec,
                         MatxTensorHandle dst, MatxTensorHandle src) {
    MATX_TRY
        visit_tensor(to_base(src), [&]<typename T, int R>(matx::tensor_t<T,R>& st) {
            visit_tensor(to_base(dst), [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) {
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::any(st)).run(ex);
                });
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_norm(MatxExecutorHandle exec,
                          MatxTensorHandle dst, MatxTensorHandle src) {
    MATX_TRY
        visit_float_tensor(to_base(src), [&]<typename T, int R>(matx::tensor_t<T,R>& st) {
            visit_float_tensor(to_base(dst), [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) {
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::vector_norm(st)).run(ex);
                });
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_trace(MatxExecutorHandle exec,
                           MatxTensorHandle dst, MatxTensorHandle src) {
    MATX_TRY
        visit_float_tensor(to_base(src), [&]<typename T, int R>(matx::tensor_t<T,R>& st) {
            visit_float_tensor(to_base(dst), [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) {
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::trace(st)).run(ex);
                });
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_cumsum(MatxExecutorHandle exec,
                            MatxTensorHandle dst, MatxTensorHandle src, int axis) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(src),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt, matx::tensor_t<T,R>& st) {
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::cumsum(st, axis)).run(ex);
                });
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_sort(MatxExecutorHandle exec,
                          MatxTensorHandle dst, MatxTensorHandle src, MatxSortDir dir) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(src),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt, matx::tensor_t<T,R>& st) {
                auto mdir = (dir == MATX_SORT_ASC) ? matx::SORT_DIR_ASC : matx::SORT_DIR_DESC;
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::sort(st, mdir)).run(ex);
                });
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}
