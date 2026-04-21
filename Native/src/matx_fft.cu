// matx_fft.cu  –  FFT, RFFT, FFTShift, DCT.
#include "dispatch.h"

// ── FFT norm conversion ───────────────────────────────────────────────────────
static matx::FFTNorm to_matx_norm(MatxFftNorm n) {
    switch (n) {
        case MATX_FFT_NORM_FORWARD:  return matx::FFTNorm::FORWARD;
        case MATX_FFT_NORM_ORTHO:    return matx::FFTNorm::ORTHO;
        default:                      return matx::FFTNorm::BACKWARD;
    }
}

// ── Complex-to-complex 1-D FFT ────────────────────────────────────────────────
extern "C" int matx_fft(MatxExecutorHandle exec,
                         MatxTensorHandle dst, MatxTensorHandle src,
                         int64_t nfft, MatxFftNorm norm) {
    MATX_TRY
        auto mn = to_matx_norm(norm);
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(to_base(src), [&]<typename ST, int SR>(matx::tensor_t<ST,SR>& st) {
                if constexpr (std::is_same_v<T, ST> && R == SR) {
                    to_exec(exec)->run([&](auto& ex) {
                        if (nfft > 0)
                            (dt = matx::fft(st, static_cast<matx::index_t>(nfft), mn)).run(ex);
                        else
                            (dt = matx::fft(st, 0, mn)).run(ex);
                    });
                } else {
                    throw std::runtime_error("fft: dst/src dtype or rank mismatch");
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_ifft(MatxExecutorHandle exec,
                          MatxTensorHandle dst, MatxTensorHandle src,
                          int64_t nfft, MatxFftNorm norm) {
    MATX_TRY
        auto mn = to_matx_norm(norm);
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(to_base(src), [&]<typename ST, int SR>(matx::tensor_t<ST,SR>& st) {
                if constexpr (std::is_same_v<T, ST> && R == SR) {
                    to_exec(exec)->run([&](auto& ex) {
                        if (nfft > 0)
                            (dt = matx::ifft(st, static_cast<matx::index_t>(nfft), mn)).run(ex);
                        else
                            (dt = matx::ifft(st, 0, mn)).run(ex);
                    });
                } else {
                    throw std::runtime_error("ifft: dst/src dtype or rank mismatch");
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── 2-D FFT ───────────────────────────────────────────────────────────────────
extern "C" int matx_fft2(MatxExecutorHandle exec,
                          MatxTensorHandle dst, MatxTensorHandle src,
                          MatxFftNorm norm) {
    MATX_TRY
        auto mn = to_matx_norm(norm);
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(to_base(src), [&]<typename ST, int SR>(matx::tensor_t<ST,SR>& st) {
                if constexpr (std::is_same_v<T, ST> && R == SR && R == 2) {
                    to_exec(exec)->run([&](auto& ex) {
                        (dt = matx::fft2(st, {0,0}, mn)).run(ex);
                    });
                } else {
                    throw std::runtime_error("fft2 requires rank-2 tensors with matching dtype");
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_ifft2(MatxExecutorHandle exec,
                           MatxTensorHandle dst, MatxTensorHandle src,
                           MatxFftNorm norm) {
    MATX_TRY
        auto mn = to_matx_norm(norm);
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(to_base(src), [&]<typename ST, int SR>(matx::tensor_t<ST,SR>& st) {
                if constexpr (std::is_same_v<T, ST> && R == SR && R == 2) {
                    to_exec(exec)->run([&](auto& ex) {
                        (dt = matx::ifft2(st, {0,0}, mn)).run(ex);
                    });
                } else {
                    throw std::runtime_error("ifft2 requires rank-2 tensors with matching dtype");
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Real-to-complex RFFT ──────────────────────────────────────────────────────
// dst must be complex, src must be real and same base precision.
extern "C" int matx_rfft(MatxExecutorHandle exec,
                          MatxTensorHandle dst, MatxTensorHandle src,
                          int64_t nfft, MatxFftNorm norm) {
    MATX_TRY
        auto mn = to_matx_norm(norm);
        auto* db = to_base(dst);
        auto* sb = to_base(src);
        // Dispatch on source dtype (float or double).
        if (sb->dtype == MATX_DTYPE_FLOAT32 && db->dtype == MATX_DTYPE_COMPLEX64) {
            visit_float_tensor(sb, [&]<typename T, int R>(matx::tensor_t<T,R>& st) {
                if constexpr (std::is_same_v<T, float>) {
                    visit_float_tensor(db, [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) {
                        if constexpr (std::is_same_v<DT, cf32_t> && R == DR) {
                            to_exec(exec)->run([&](auto& ex) {
                                auto n = nfft > 0 ? static_cast<matx::index_t>(nfft) : 0;
                                (dt = matx::rfft(st, n, mn)).run(ex);
                            });
                        }
                    });
                }
            });
        } else if (sb->dtype == MATX_DTYPE_FLOAT64 && db->dtype == MATX_DTYPE_COMPLEX128) {
            visit_float_tensor(sb, [&]<typename T, int R>(matx::tensor_t<T,R>& st) {
                if constexpr (std::is_same_v<T, double>) {
                    visit_float_tensor(db, [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) {
                        if constexpr (std::is_same_v<DT, cd64_t> && R == DR) {
                            to_exec(exec)->run([&](auto& ex) {
                                auto n = nfft > 0 ? static_cast<matx::index_t>(nfft) : 0;
                                (dt = matx::rfft(st, n, mn)).run(ex);
                            });
                        }
                    });
                }
            });
        } else {
            throw std::runtime_error("rfft: src must be float32/float64, dst must be complex64/complex128");
        }
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_irfft(MatxExecutorHandle exec,
                           MatxTensorHandle dst, MatxTensorHandle src,
                           int64_t nfft, MatxFftNorm norm) {
    MATX_TRY
        auto mn = to_matx_norm(norm);
        auto* db = to_base(dst);
        auto* sb = to_base(src);
        if (sb->dtype == MATX_DTYPE_COMPLEX64 && db->dtype == MATX_DTYPE_FLOAT32) {
            visit_float_tensor(sb, [&]<typename T, int R>(matx::tensor_t<T,R>& st) {
                if constexpr (std::is_same_v<T, cf32_t>) {
                    visit_float_tensor(db, [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) {
                        if constexpr (std::is_same_v<DT, float> && R == DR) {
                            to_exec(exec)->run([&](auto& ex) {
                                auto n = nfft > 0 ? static_cast<matx::index_t>(nfft) : 0;
                                (dt = matx::irfft(st, n, mn)).run(ex);
                            });
                        }
                    });
                }
            });
        } else if (sb->dtype == MATX_DTYPE_COMPLEX128 && db->dtype == MATX_DTYPE_FLOAT64) {
            visit_float_tensor(sb, [&]<typename T, int R>(matx::tensor_t<T,R>& st) {
                if constexpr (std::is_same_v<T, cd64_t>) {
                    visit_float_tensor(db, [&]<typename DT, int DR>(matx::tensor_t<DT,DR>& dt) {
                        if constexpr (std::is_same_v<DT, double> && R == DR) {
                            to_exec(exec)->run([&](auto& ex) {
                                auto n = nfft > 0 ? static_cast<matx::index_t>(nfft) : 0;
                                (dt = matx::irfft(st, n, mn)).run(ex);
                            });
                        }
                    });
                }
            });
        } else {
            throw std::runtime_error("irfft: src must be complex64/complex128, dst must be float32/float64");
        }
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── FFTShift ──────────────────────────────────────────────────────────────────
extern "C" int matx_fftshift(MatxExecutorHandle exec,
                              MatxTensorHandle dst, MatxTensorHandle src) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(src),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt, matx::tensor_t<T,R>& st) {
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::fftshift(st)).run(ex);
                });
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

extern "C" int matx_ifftshift(MatxExecutorHandle exec,
                               MatxTensorHandle dst, MatxTensorHandle src) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(src),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt, matx::tensor_t<T,R>& st) {
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::ifftshift(st)).run(ex);
                });
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── DCT ───────────────────────────────────────────────────────────────────────
extern "C" int matx_dct(MatxExecutorHandle exec,
                         MatxTensorHandle dst, MatxTensorHandle src) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(src),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt, matx::tensor_t<T,R>& st) {
                to_exec(exec)->run([&](auto& ex) {
                    (dt = matx::dct(st)).run(ex);
                });
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}
