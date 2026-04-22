// tensor_handle.h  –  Internal type-erased tensor wrapper.
// This file is only included by the .cu implementation files.
#pragma once

#include <matx.h>
#include <stdexcept>
#include <string>
#include <cstring>
#include "matx_api.h"

// ── Convenient type aliases ──────────────────────────────────────────────────
using cf32_t = cuda::std::complex<float>;
using cd64_t = cuda::std::complex<double>;

// ── Memory space conversion ──────────────────────────────────────────────────
inline matx::matxMemorySpace_t to_matx_mem(MatxMemorySpace s) noexcept {
    switch (s) {
        case MATX_MEMORY_MANAGED:     return matx::MATX_MANAGED_MEMORY;
        case MATX_MEMORY_DEVICE:      return matx::MATX_DEVICE_MEMORY;
        case MATX_MEMORY_HOST:        return matx::MATX_HOST_MEMORY;
        case MATX_MEMORY_HOST_MALLOC: return matx::MATX_HOST_MALLOC_MEMORY;
        default:                      return matx::MATX_MANAGED_MEMORY;
    }
}

// ── dtype ↔ sizeof ───────────────────────────────────────────────────────────
inline size_t dtype_sizeof(MatxDtype d) noexcept {
    switch (d) {
        case MATX_DTYPE_FLOAT32:    return 4;
        case MATX_DTYPE_FLOAT64:    return 8;
        case MATX_DTYPE_COMPLEX64:  return 8;
        case MATX_DTYPE_COMPLEX128: return 16;
        case MATX_DTYPE_INT32:      return 4;
        case MATX_DTYPE_INT64:      return 8;
        case MATX_DTYPE_UINT32:     return 4;
        case MATX_DTYPE_UINT8:      return 1;
        default:                    return 0;
    }
}

template<typename T> constexpr MatxDtype dtype_of() noexcept;
template<> constexpr MatxDtype dtype_of<float>()    noexcept { return MATX_DTYPE_FLOAT32; }
template<> constexpr MatxDtype dtype_of<double>()   noexcept { return MATX_DTYPE_FLOAT64; }
template<> constexpr MatxDtype dtype_of<cf32_t>()   noexcept { return MATX_DTYPE_COMPLEX64; }
template<> constexpr MatxDtype dtype_of<cd64_t>()   noexcept { return MATX_DTYPE_COMPLEX128; }
template<> constexpr MatxDtype dtype_of<int32_t>()  noexcept { return MATX_DTYPE_INT32; }
template<> constexpr MatxDtype dtype_of<int64_t>()  noexcept { return MATX_DTYPE_INT64; }
template<> constexpr MatxDtype dtype_of<uint32_t>() noexcept { return MATX_DTYPE_UINT32; }
template<> constexpr MatxDtype dtype_of<uint8_t>()  noexcept { return MATX_DTYPE_UINT8; }

// ── Abstract base ────────────────────────────────────────────────────────────
struct MatxTensorBase {
    MatxDtype dtype{};
    int       rank{};

    virtual ~MatxTensorBase() = default;

    virtual void*   data_ptr()            = 0;
    virtual int64_t size  (int dim) const = 0;
    virtual int64_t stride(int dim) const = 0;
    virtual int64_t total_size()    const = 0;
    virtual void    tensor_print()  const = 0;
    virtual void    set_name(const char* n) = 0;
    virtual void    prefetch(cudaStream_t stream) = 0;
};

// ── Typed concrete wrapper ───────────────────────────────────────────────────
template<typename T, int RANK>
struct MatxTypedTensor final : MatxTensorBase {
    matx::tensor_t<T, RANK> t;

    /// Owning construction from a C int64_t shape array.
    MatxTypedTensor(const int64_t* shape, MatxMemorySpace mem) {
        dtype = dtype_of<T>();
        rank  = RANK;
        matx::index_t dims[RANK];
        for (int i = 0; i < RANK; ++i) dims[i] = static_cast<matx::index_t>(shape[i]);
        matx::make_tensor(t, dims, to_matx_mem(mem));
    }

    /// Non-owning view over an existing pointer.
    MatxTypedTensor(void* ptr, const int64_t* shape) {
        dtype = dtype_of<T>();
        rank  = RANK;
        matx::index_t dims[RANK];
        for (int i = 0; i < RANK; ++i) dims[i] = static_cast<matx::index_t>(shape[i]);
        t = matx::make_tensor<T>(static_cast<T*>(ptr), dims, /*owning=*/false);
    }

    /// Adopt an already-constructed tensor_t (used for view results like Permute).
    explicit MatxTypedTensor(matx::tensor_t<T, RANK> existing)
        : t(std::move(existing))
    {
        dtype = dtype_of<T>();
        rank  = RANK;
    }

    void*   data_ptr()           override { return static_cast<void*>(t.Data()); }
    int64_t size  (int d) const  override { return static_cast<int64_t>(t.Size(d)); }
    int64_t stride(int d) const  override { return static_cast<int64_t>(t.Stride(d)); }
    int64_t total_size()  const  override {
        int64_t n = 1;
        for (int i = 0; i < RANK; ++i) n *= t.Size(i);
        return n;
    }
    void tensor_print() const override { matx::print(t); }
    void set_name(const char* n) override { t.set_name(n); }
    void prefetch(cudaStream_t stream) override {
        t.PrefetchDevice(stream);
    }
};
