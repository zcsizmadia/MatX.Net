// matx_generators.cu  –  Fill tensors with generated sequences / windows.
#include "dispatch.h"

// ── zeros / ones ──────────────────────────────────────────────────────────────

extern "C" int matx_fill_zeros(MatxExecutorHandle exec, MatxTensorHandle dst) {
    MATX_TRY
        visit_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            to_exec(exec)->run([&](auto& ex) {
                (dt = matx::zeros<T>(dt.Shape())).run(ex);
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_fill_ones(MatxExecutorHandle exec, MatxTensorHandle dst) {
    MATX_TRY
        visit_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            to_exec(exec)->run([&](auto& ex) {
                (dt = matx::ones<T>(dt.Shape())).run(ex);
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── random ────────────────────────────────────────────────────────────────────

extern "C" int matx_fill_random_uniform(MatxExecutorHandle exec, MatxTensorHandle dst) {
    MATX_TRY
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            to_exec(exec)->run([&](auto& ex) {
                (dt = matx::random<T>(dt.Shape(), matx::UNIFORM)).run(ex);
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_fill_random_normal(MatxExecutorHandle exec, MatxTensorHandle dst) {
    MATX_TRY
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            to_exec(exec)->run([&](auto& ex) {
                (dt = matx::random<T>(dt.Shape(), matx::NORMAL)).run(ex);
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── linspace / range ──────────────────────────────────────────────────────────

extern "C" int matx_fill_linspace(MatxExecutorHandle exec, MatxTensorHandle dst,
                                   double start, double stop) {
    MATX_TRY
        auto* base = to_base(dst);
        if (base->rank != 1)
            throw std::runtime_error("linspace requires a 1-D tensor");
        visit_float_tensor(base, [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            if constexpr (R == 1) {
                T s0 = static_cast<T>(start);
                T s1 = static_cast<T>(stop);
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::linspace<T>(s0, s1, dt.Size(0))).run(ex);
                });
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_fill_range(MatxExecutorHandle exec, MatxTensorHandle dst,
                                double start, double step) {
    MATX_TRY
        auto* base = to_base(dst);
        if (base->rank != 1)
            throw std::runtime_error("range requires a 1-D tensor");
        visit_float_tensor(base, [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            if constexpr (R == 1) {
                T s0   = static_cast<T>(start);
                T step_ = static_cast<T>(step);
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::range<T>(s0, step_, dt.Size(0))).run(ex);
                });
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── fftfreq ───────────────────────────────────────────────────────────────────

extern "C" int matx_fill_fftfreq(MatxExecutorHandle exec, MatxTensorHandle dst,
                                  double sample_rate) {
    MATX_TRY
        auto* base = to_base(dst);
        if (base->rank != 1)
            throw std::runtime_error("fftfreq requires a 1-D tensor");
        visit_float_tensor(base, [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            if constexpr (R == 1) {
                T fs = static_cast<T>(sample_rate);
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::fftfreq(dt.Size(0), fs)).run(ex);
                });
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Window functions (1-D real only) ─────────────────────────────────────────
// MatX window generators are parameterised at compile time on
//   <Axis, TotalRank, T>. For 1-D fills we always use Axis=0, Rank=1.

#define IMPL_WINDOW(fn_name, gen_name)                                          \
extern "C" int fn_name(MatxExecutorHandle exec, MatxTensorHandle dst) {         \
    MATX_TRY                                                                    \
        auto* base = to_base(dst);                                              \
        if (base->rank != 1)                                                    \
            throw std::runtime_error(#fn_name " requires a 1-D tensor");        \
        visit_float_tensor(base, [&]<typename T, int R>(matx::tensor_t<T,R>& dt) { \
            if constexpr (R == 1) {                                             \
                to_exec(exec)->run([&](auto& ex) {                              \
                    (dt = matx::gen_name<0, 1, T>(dt.Shape())).run(ex);        \
                });                                                             \
            }                                                                   \
        });                                                                     \
        return MATX_OK;                                                         \
    MATX_CATCH(MATX_ERR_TYPE)                                                   \
}

IMPL_WINDOW(matx_fill_hamming,  hamming)
IMPL_WINDOW(matx_fill_hanning,  hanning)
IMPL_WINDOW(matx_fill_blackman, blackman)
IMPL_WINDOW(matx_fill_bartlett, bartlett)

#undef IMPL_WINDOW
