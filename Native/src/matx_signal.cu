// matx_signal.cu  –  Convolution, correlation, Pwelch, resample_poly.
#include "dispatch.h"

// ── Convolution mode ──────────────────────────────────────────────────────────
static matx::matxConvCorrMode_t to_matx_conv(MatxConvMode m) {
    switch (m) {
        case MATX_CONV_SAME:  return matx::MATX_C_MODE_SAME;
        case MATX_CONV_VALID: return matx::MATX_C_MODE_VALID;
        default:              return matx::MATX_C_MODE_FULL;
    }
}

// ── 1-D Convolution ───────────────────────────────────────────────────────────
// signal: (..., N)  filter: (K,)  dst: (..., M)  where M depends on mode.
extern "C" int matx_conv1d(MatxExecutorHandle exec,
                            MatxTensorHandle dst,
                            MatxTensorHandle signal, MatxTensorHandle filter,
                            MatxConvMode mode) {
    MATX_TRY
        auto cm = to_matx_conv(mode);
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(to_base(signal), [&]<typename TS, int RS>(matx::tensor_t<TS,RS>& st) {
                if constexpr (std::is_same_v<T,TS> && R == RS) {
                    visit_float_tensor(to_base(filter), [&]<typename TF, int RF>(matx::tensor_t<TF,RF>& ft) {
                        if constexpr (std::is_same_v<T,TF> && RF == 1) {
                            to_exec(exec)->run([&](auto& ex) {
                                (dt = matx::conv1d(st, ft, cm)).run(ex);
                            });
                        } else {
                            throw std::runtime_error("conv1d: filter must be rank-1 with same dtype as signal");
                        }
                    });
                } else {
                    throw std::runtime_error("conv1d: dst and signal rank/dtype must match");
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── 2-D Convolution ───────────────────────────────────────────────────────────
extern "C" int matx_conv2d(MatxExecutorHandle exec,
                            MatxTensorHandle dst,
                            MatxTensorHandle signal, MatxTensorHandle filter,
                            MatxConvMode mode) {
    MATX_TRY
        auto cm = to_matx_conv(mode);
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(to_base(signal), [&]<typename TS, int RS>(matx::tensor_t<TS,RS>& st) {
                if constexpr (std::is_same_v<T,TS> && R == RS) {
                    visit_float_tensor(to_base(filter), [&]<typename TF, int RF>(matx::tensor_t<TF,RF>& ft) {
                        if constexpr (std::is_same_v<T,TF> && RF == 2) {
                            to_exec(exec)->run([&](auto& ex) {
                                (dt = matx::conv2d(st, ft, cm)).run(ex);
                            });
                        } else {
                            throw std::runtime_error("conv2d: filter must be rank-2 with same dtype as signal");
                        }
                    });
                } else {
                    throw std::runtime_error("conv2d: dst and signal rank/dtype must match");
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Cross-correlation ─────────────────────────────────────────────────────────
extern "C" int matx_corr(MatxExecutorHandle exec,
                          MatxTensorHandle dst,
                          MatxTensorHandle a, MatxTensorHandle b) {
    MATX_TRY
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(to_base(a), [&]<typename TA, int RA>(matx::tensor_t<TA,RA>& at) {
                if constexpr (std::is_same_v<T,TA> && R == RA) {
                    visit_float_tensor(to_base(b), [&]<typename TB, int RB>(matx::tensor_t<TB,RB>& bt) {
                        if constexpr (std::is_same_v<T,TB> && R == RB) {
                            to_exec(exec)->run([&](auto& ex) {
                                (dt = matx::corr(at, bt, matx::MATX_C_MODE_FULL)).run(ex);
                            });
                        }
                    });
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Welch PSD ─────────────────────────────────────────────────────────────────
// pxx:    (batch, nfft/2+1)  float
// signal: (batch, N)         float
// window: (win_size,)        float
extern "C" int matx_pwelch(MatxExecutorHandle exec,
                            MatxTensorHandle pxx,
                            MatxTensorHandle signal,
                            MatxTensorHandle window,
                            int64_t noverlap,
                            int64_t nfft) {
    MATX_TRY
        // Only float32 supported for now (Welch is most commonly float).
        if (to_base(signal)->dtype != MATX_DTYPE_FLOAT32)
            throw std::runtime_error("pwelch: signal must be float32");

        visit_float_tensor(to_base(pxx), [&]<typename TP, int RP>(matx::tensor_t<TP,RP>& pt) {
            if constexpr (std::is_same_v<TP, float> && RP == 2) {
                visit_float_tensor(to_base(signal), [&]<typename TS, int RS>(matx::tensor_t<TS,RS>& st) {
                    if constexpr (std::is_same_v<TS, float> && RS == 2) {
                        visit_float_tensor(to_base(window), [&]<typename TW, int RW>(matx::tensor_t<TW,RW>& wt) {
                            if constexpr (std::is_same_v<TW, float> && RW == 1) {
                                to_exec(exec)->run([&](auto& ex) {
                                    (pt = matx::pwelch(st, wt,
                                        static_cast<matx::index_t>(noverlap),
                                        static_cast<matx::index_t>(nfft))).run(ex);
                                });
                            } else {
                                throw std::runtime_error("pwelch: window must be 1-D float32");
                            }
                        });
                    } else {
                        throw std::runtime_error("pwelch: signal must be 2-D float32");
                    }
                });
            } else {
                throw std::runtime_error("pwelch: pxx must be 2-D float32");
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Polyphase resample ────────────────────────────────────────────────────────
// src: (N,)  dst: (M,)  filter: (K,) optional (nullptr = default)
extern "C" int matx_resample_poly(MatxExecutorHandle exec,
                                   MatxTensorHandle dst,
                                   MatxTensorHandle src,
                                   int up, int down,
                                   MatxTensorHandle filter) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(src),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt, matx::tensor_t<T,R>& st) {
                if constexpr (R == 1 || R == 2) {
                    to_exec(exec)->run([&](auto& ex) {
                        if (filter) {
                            visit_float_tensor(to_base(filter),
                                [&]<typename TF, int RF>(matx::tensor_t<TF,RF>& ft) {
                                    if constexpr (std::is_same_v<T,TF> && RF == 1) {
                                        (dt = matx::resample_poly(st, up, down, ft)).run(ex);
                                    } else {
                                        throw std::runtime_error(
                                            "resample_poly: filter must be 1-D with same dtype");
                                    }
                                });
                        } else {
                            (dt = matx::resample_poly(st, up, down)).run(ex);
                        }
                    });
                } else {
                    throw std::runtime_error("resample_poly: tensors must be rank-1 or rank-2");
                }
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}
