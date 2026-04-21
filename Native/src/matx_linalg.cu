// matx_linalg.cu  –  Matrix multiply, decompositions, solvers.
#include "dispatch.h"

// ── matmul ────────────────────────────────────────────────────────────────────
// Supports batched matmul: rank-3 tensors treated as (batch, M, N).
extern "C" int matx_matmul(MatxExecutorHandle exec,
                            MatxTensorHandle dst,
                            MatxTensorHandle a, MatxTensorHandle b) {
    MATX_TRY
        auto* db = to_base(dst);
        auto* ab = to_base(a);
        auto* bb = to_base(b);
        if (db->dtype != ab->dtype || ab->dtype != bb->dtype)
            throw std::runtime_error("matmul: dtype mismatch");

        visit_float_tensor(db, [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(ab, [&]<typename TA, int RA>(matx::tensor_t<TA,RA>& at) {
                visit_float_tensor(bb, [&]<typename TB, int RB>(matx::tensor_t<TB,RB>& bt) {
                    if constexpr (std::is_same_v<T,TA> && std::is_same_v<T,TB>
                                  && R == RA && R == RB && (R == 2 || R == 3)) {
                        to_exec(exec)->run([&](auto& ex) {
                            (dt = matx::matmul(at, bt)).run(ex);
                        });
                    } else {
                        throw std::runtime_error("matmul: requires rank-2 or rank-3, same dtype");
                    }
                });
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── matvec ────────────────────────────────────────────────────────────────────
extern "C" int matx_matvec(MatxExecutorHandle exec,
                            MatxTensorHandle dst,
                            MatxTensorHandle a, MatxTensorHandle x) {
    MATX_TRY
        auto* ab = to_base(a);
        auto* xb = to_base(x);
        auto* db = to_base(dst);
        if (ab->rank != 2 || xb->rank != 1 || db->rank != 1)
            throw std::runtime_error("matvec: a must be rank-2, x and dst must be rank-1");

        visit_float_tensor(db, [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            if constexpr (R == 1) {
                visit_float_tensor(ab, [&]<typename TA, int RA>(matx::tensor_t<TA,RA>& at) {
                    if constexpr (std::is_same_v<T,TA> && RA == 2) {
                        visit_float_tensor(xb, [&]<typename TX, int RX>(matx::tensor_t<TX,RX>& xt) {
                            if constexpr (std::is_same_v<T,TX> && RX == 1) {
                                to_exec(exec)->run([&](auto& ex) {
                                    (dt = matx::matvec(at, xt)).run(ex);
                                });
                            }
                        });
                    }
                });
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── outer ─────────────────────────────────────────────────────────────────────
extern "C" int matx_outer(MatxExecutorHandle exec,
                           MatxTensorHandle dst,
                           MatxTensorHandle a, MatxTensorHandle b) {
    MATX_TRY
        auto* ab = to_base(a);
        auto* bb = to_base(b);
        auto* db = to_base(dst);
        if (ab->rank != 1 || bb->rank != 1 || db->rank != 2)
            throw std::runtime_error("outer: a and b must be rank-1, dst must be rank-2");

        visit_float_tensor(db, [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            if constexpr (R == 2) {
                visit_float_tensor(ab, [&]<typename TA, int RA>(matx::tensor_t<TA,RA>& at) {
                    if constexpr (std::is_same_v<T,TA> && RA == 1) {
                        visit_float_tensor(bb, [&]<typename TB, int RB>(matx::tensor_t<TB,RB>& bt) {
                            if constexpr (std::is_same_v<T,TB> && RB == 1) {
                                to_exec(exec)->run([&](auto& ex) {
                                    (dt = matx::outer(at, bt)).run(ex);
                                });
                            }
                        });
                    }
                });
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── SVD ───────────────────────────────────────────────────────────────────────
// a: (M, N)  u: (M, K)  s: (K,)  vt: (K, N)
// For batched: (batch, M, N) etc.
extern "C" int matx_svd(MatxExecutorHandle exec,
                         MatxTensorHandle u, MatxTensorHandle s, MatxTensorHandle vt,
                         MatxTensorHandle a, MatxSvdMode mode) {
    MATX_TRY
        auto svd_mode = (mode == MATX_SVD_FULL) ? matx::SVDMode::FULL :
                        (mode == MATX_SVD_NONE) ? matx::SVDMode::NONE :
                                                   matx::SVDMode::REDUCED;
        visit_float_tensor(to_base(a), [&]<typename T, int R>(matx::tensor_t<T,R>& at) {
            if constexpr (R == 2 || R == 3) {
                visit_float_tensor(to_base(u), [&]<typename TU, int RU>(matx::tensor_t<TU,RU>& ut) {
                    if constexpr (std::is_same_v<T,TU> && R == RU) {
                        visit_float_tensor(to_base(s), [&]<typename TS, int RS>(matx::tensor_t<TS,RS>& st) {
                            // s has rank 1 (or rank R-1 for batched)
                            visit_float_tensor(to_base(vt), [&]<typename TV, int RV>(matx::tensor_t<TV,RV>& vtt) {
                                if constexpr (std::is_same_v<T,TV> && R == RV) {
                                    to_exec(exec)->run([&](auto& ex) {
                                        (matx::mtie(ut, st, vtt) = matx::svd(at, svd_mode)).run(ex);
                                    });
                                }
                            });
                        });
                    }
                });
            } else {
                throw std::runtime_error("svd: input must be rank-2 or rank-3");
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── SVD power iteration ───────────────────────────────────────────────────────
// a: (M, N)  u: (M, k)  s: (k,)  vt: (k, N)
// x0: initial random matrix (M, k)
extern "C" int matx_svdpi(MatxExecutorHandle exec,
                           MatxTensorHandle u, MatxTensorHandle s, MatxTensorHandle vt,
                           MatxTensorHandle a, int iterations, int k) {
    MATX_TRY
        visit_float_tensor(to_base(a), [&]<typename T, int R>(matx::tensor_t<T,R>& at) {
            if constexpr (R == 2) {
                visit_float_tensor(to_base(u), [&]<typename TU, int RU>(matx::tensor_t<TU,RU>& ut) {
                    if constexpr (std::is_same_v<T,TU> && RU == 2) {
                        visit_float_tensor(to_base(s), [&]<typename TS, int RS>(matx::tensor_t<TS,RS>& st) {
                            if constexpr (RS == 1) {
                                visit_float_tensor(to_base(vt), [&]<typename TV, int RV>(matx::tensor_t<TV,RV>& vtt) {
                                    if constexpr (std::is_same_v<T,TV> && RV == 2) {
                                        // Create random initial guess (M x k)
                                        int64_t shape[2] = { at.Size(0), (int64_t)k };
                                        auto x0 = matx::make_tensor<T>(
                                            cuda::std::array<matx::index_t,2>{at.Size(0), (matx::index_t)k});
                                        (x0 = matx::random<T>(x0.Shape(), matx::UNIFORM)).run(
                                            to_exec(exec)->kind == ExecutorHandle::Kind::CUDA
                                                ? to_exec(exec)->cuda_exec
                                                : static_cast<matx::cudaExecutor>(matx::cudaExecutor{}));
                                        to_exec(exec)->run([&](auto& ex) {
                                            (matx::mtie(ut, st, vtt) =
                                                matx::svdpi(at, x0, iterations, k)).run(ex);
                                        });
                                    }
                                });
                            }
                        });
                    }
                });
            } else {
                throw std::runtime_error("svdpi: input must be rank-2");
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── QR ────────────────────────────────────────────────────────────────────────
extern "C" int matx_qr(MatxExecutorHandle exec,
                        MatxTensorHandle q, MatxTensorHandle r,
                        MatxTensorHandle a) {
    MATX_TRY
        visit_float_tensor(to_base(a), [&]<typename T, int R>(matx::tensor_t<T,R>& at) {
            if constexpr (R == 2 || R == 3) {
                visit_float_tensor(to_base(q), [&]<typename TQ, int RQ>(matx::tensor_t<TQ,RQ>& qt) {
                    if constexpr (std::is_same_v<T,TQ> && R == RQ) {
                        visit_float_tensor(to_base(r), [&]<typename TR, int RR>(matx::tensor_t<TR,RR>& rt) {
                            if constexpr (std::is_same_v<T,TR> && R == RR) {
                                to_exec(exec)->run([&](auto& ex) {
                                    (matx::mtie(qt, rt) = matx::qr(at)).run(ex);
                                });
                            }
                        });
                    }
                });
            } else {
                throw std::runtime_error("qr: input must be rank-2 or rank-3");
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── LU ────────────────────────────────────────────────────────────────────────
// l: (M,M)  u_out: (M,N)  piv: (M,) int  a: (M,N)
extern "C" int matx_lu(MatxExecutorHandle exec,
                        MatxTensorHandle l, MatxTensorHandle u_out, MatxTensorHandle piv,
                        MatxTensorHandle a) {
    MATX_TRY
        visit_float_tensor(to_base(a), [&]<typename T, int R>(matx::tensor_t<T,R>& at) {
            if constexpr (R == 2 || R == 3) {
                visit_float_tensor(to_base(l), [&]<typename TL, int RL>(matx::tensor_t<TL,RL>& lt) {
                    if constexpr (std::is_same_v<T,TL> && R == RL) {
                        visit_float_tensor(to_base(u_out), [&]<typename TU, int RU>(matx::tensor_t<TU,RU>& ut) {
                            if constexpr (std::is_same_v<T,TU> && R == RU) {
                                // pivot tensor – int type
                                visit_tensor(to_base(piv), [&]<typename TP, int RP>(matx::tensor_t<TP,RP>& pt) {
                                    to_exec(exec)->run([&](auto& ex) {
                                        (matx::mtie(lt, ut, pt) = matx::lu(at)).run(ex);
                                    });
                                });
                            }
                        });
                    }
                });
            } else {
                throw std::runtime_error("lu: input must be rank-2 or rank-3");
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Cholesky ──────────────────────────────────────────────────────────────────
extern "C" int matx_chol(MatxExecutorHandle exec,
                          MatxTensorHandle l,
                          MatxTensorHandle a) {
    MATX_TRY
        visit_tensor2(to_base(l), to_base(a),
            [&]<typename T, int R>(matx::tensor_t<T,R>& lt, matx::tensor_t<T,R>& at) {
                if constexpr (R == 2 || R == 3) {
                    to_exec(exec)->run([&](auto& ex) {
                        (lt = matx::chol(at)).run(ex);
                    });
                } else {
                    throw std::runtime_error("chol: input must be rank-2 or rank-3");
                }
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Eigen decomposition ───────────────────────────────────────────────────────
// vectors: (N,N) complex, values: (N,) complex
extern "C" int matx_eig(MatxExecutorHandle exec,
                         MatxTensorHandle vectors, MatxTensorHandle values,
                         MatxTensorHandle a) {
    MATX_TRY
        visit_float_tensor(to_base(a), [&]<typename T, int R>(matx::tensor_t<T,R>& at) {
            if constexpr (R == 2 || R == 3) {
                visit_float_tensor(to_base(vectors), [&]<typename TV, int RV>(matx::tensor_t<TV,RV>& vt) {
                    if constexpr (R == RV) {
                        visit_float_tensor(to_base(values), [&]<typename TW, int RW>(matx::tensor_t<TW,RW>& wt) {
                            to_exec(exec)->run([&](auto& ex) {
                                (matx::mtie(vt, wt) = matx::eig(at)).run(ex);
                            });
                        });
                    }
                });
            } else {
                throw std::runtime_error("eig: input must be rank-2 or rank-3");
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Linear system solve ───────────────────────────────────────────────────────
extern "C" int matx_solve(MatxExecutorHandle exec,
                           MatxTensorHandle x,
                           MatxTensorHandle a, MatxTensorHandle b) {
    MATX_TRY
        visit_float_tensor(to_base(x), [&]<typename T, int R>(matx::tensor_t<T,R>& xt) {
            visit_float_tensor(to_base(a), [&]<typename TA, int RA>(matx::tensor_t<TA,RA>& at) {
                if constexpr (std::is_same_v<T,TA>) {
                    visit_float_tensor(to_base(b), [&]<typename TB, int RB>(matx::tensor_t<TB,RB>& bt) {
                        if constexpr (std::is_same_v<T,TB>) {
                            to_exec(exec)->run([&](auto& ex) {
                                (xt = matx::solve(at, bt)).run(ex);
                            });
                        }
                    });
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Inverse ───────────────────────────────────────────────────────────────────
extern "C" int matx_inv(MatxExecutorHandle exec,
                         MatxTensorHandle dst, MatxTensorHandle a) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(a),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt, matx::tensor_t<T,R>& at) {
                if constexpr (R == 2 || R == 3) {
                    to_exec(exec)->run([&](auto& ex) {
                        (dt = matx::inv(at)).run(ex);
                    });
                } else {
                    throw std::runtime_error("inv: input must be rank-2 or rank-3");
                }
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Pseudo-inverse ────────────────────────────────────────────────────────────
extern "C" int matx_pinv(MatxExecutorHandle exec,
                          MatxTensorHandle dst, MatxTensorHandle a) {
    MATX_TRY
        visit_tensor2(to_base(dst), to_base(a),
            [&]<typename T, int R>(matx::tensor_t<T,R>& dt, matx::tensor_t<T,R>& at) {
                if constexpr (R == 2 || R == 3) {
                    to_exec(exec)->run([&](auto& ex) {
                        (dt = matx::pinv(at)).run(ex);
                    });
                } else {
                    throw std::runtime_error("pinv: input must be rank-2 or rank-3");
                }
            });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Determinant ───────────────────────────────────────────────────────────────
extern "C" int matx_det(MatxExecutorHandle exec,
                         MatxTensorHandle dst, MatxTensorHandle a) {
    MATX_TRY
        visit_float_tensor(to_base(a), [&]<typename T, int R>(matx::tensor_t<T,R>& at) {
            if constexpr (R == 2 || R == 3) {
                visit_float_tensor(to_base(dst), [&]<typename TD, int RD>(matx::tensor_t<TD,RD>& dt) {
                    to_exec(exec)->run([&](auto& ex) {
                        (dt = matx::det(at)).run(ex);
                    });
                });
            } else {
                throw std::runtime_error("det: input must be rank-2 or rank-3");
            }
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}

// ── Einstein summation (two operands) ─────────────────────────────────────────
extern "C" int matx_einsum2(MatxExecutorHandle exec, MatxTensorHandle dst,
                             const char* subscripts,
                             MatxTensorHandle a, MatxTensorHandle b) {
    MATX_TRY
        visit_float_tensor(to_base(dst), [&]<typename T, int R>(matx::tensor_t<T,R>& dt) {
            visit_float_tensor(to_base(a), [&]<typename TA, int RA>(matx::tensor_t<TA,RA>& at) {
                if constexpr (std::is_same_v<T,TA>) {
                    visit_float_tensor(to_base(b), [&]<typename TB, int RB>(matx::tensor_t<TB,RB>& bt) {
                        if constexpr (std::is_same_v<T,TB>) {
                            to_exec(exec)->run([&](auto& ex) {
                                (dt = matx::einsum(subscripts, at, bt)).run(ex);
                            });
                        }
                    });
                }
            });
        });
        return MATX_OK;
    MATX_CATCH(MATX_ERR_TYPE)
}
