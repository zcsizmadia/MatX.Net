// matx_core.cu  –  Tensor creation, destruction, metadata, executor management.
#include "dispatch.h"
#include <cstring>
#include <atomic>

// ── Thread-local error storage ───────────────────────────────────────────────
static thread_local std::string tl_last_error;

extern "C" void matx_set_last_error(const char* msg) {
    tl_last_error = msg ? msg : "";
}

extern "C" const char* matx_get_last_error(void) {
    return tl_last_error.c_str();
}

extern "C" void matx_clear_error(void) {
    tl_last_error.clear();
}

// ── Tensor factory ────────────────────────────────────────────────────────────

// Dispatch table: create a typed tensor for the given dtype+rank
static MatxTensorBase* create_typed(MatxDtype dtype, int rank,
                                    const int64_t* shape, MatxMemorySpace mem) {
#define MAKE(T, DENUM)                                                          \
    if (dtype == DENUM) {                                                       \
        switch (rank) {                                                         \
            case 1: return new MatxTypedTensor<T,1>(shape, mem);                \
            case 2: return new MatxTypedTensor<T,2>(shape, mem);                \
            case 3: return new MatxTypedTensor<T,3>(shape, mem);                \
            case 4: return new MatxTypedTensor<T,4>(shape, mem);                \
            default: throw std::runtime_error("rank > 4 not supported");        \
        }                                                                       \
    }
    MATX_FOREACH_DTYPE(MAKE)
#undef MAKE
    throw std::runtime_error("unsupported dtype");
}

static MatxTensorBase* create_view(MatxDtype dtype, int rank,
                                   const int64_t* shape, void* ptr) {
#define MAKE(T, DENUM)                                                          \
    if (dtype == DENUM) {                                                       \
        switch (rank) {                                                         \
            case 1: return new MatxTypedTensor<T,1>(ptr, shape);                \
            case 2: return new MatxTypedTensor<T,2>(ptr, shape);                \
            case 3: return new MatxTypedTensor<T,3>(ptr, shape);                \
            case 4: return new MatxTypedTensor<T,4>(ptr, shape);                \
            default: throw std::runtime_error("rank > 4 not supported");        \
        }                                                                       \
    }
    MATX_FOREACH_DTYPE(MAKE)
#undef MAKE
    throw std::runtime_error("unsupported dtype");
}

// ── Public C API ──────────────────────────────────────────────────────────────

extern "C" MatxTensorHandle matx_tensor_create(
        MatxDtype dtype, int rank, const int64_t* shape, MatxMemorySpace mem) {
    MATX_TRY
        return create_typed(dtype, rank, shape, mem);
    MATX_CATCH(nullptr)
}

extern "C" MatxTensorHandle matx_tensor_from_ptr(
        MatxDtype dtype, int rank, const int64_t* shape, void* data) {
    MATX_TRY
        return create_view(dtype, rank, shape, data);
    MATX_CATCH(nullptr)
}

extern "C" void matx_tensor_destroy(MatxTensorHandle h) {
    delete to_base(h);
}

extern "C" int matx_tensor_rank(MatxTensorHandle h) {
    return to_base(h)->rank;
}

extern "C" MatxDtype matx_tensor_dtype(MatxTensorHandle h) {
    return to_base(h)->dtype;
}

extern "C" int64_t matx_tensor_size(MatxTensorHandle h, int dim) {
    return to_base(h)->size(dim);
}

extern "C" int64_t matx_tensor_stride(MatxTensorHandle h, int dim) {
    return to_base(h)->stride(dim);
}

extern "C" int64_t matx_tensor_total_size(MatxTensorHandle h) {
    return to_base(h)->total_size();
}

extern "C" void* matx_tensor_data(MatxTensorHandle h) {
    return to_base(h)->data_ptr();
}

extern "C" void matx_tensor_set_name(MatxTensorHandle h, const char* name) {
    to_base(h)->set_name(name);
}

extern "C" void matx_tensor_print(MatxTensorHandle h) {
    to_base(h)->tensor_print();
}

// ── Executor ──────────────────────────────────────────────────────────────────

extern "C" MatxExecutorHandle matx_executor_cuda(void* stream) {
    MATX_TRY
        return new ExecutorHandle(reinterpret_cast<cudaStream_t>(stream));
    MATX_CATCH(nullptr)
}

extern "C" MatxExecutorHandle matx_executor_host(void) {
    MATX_TRY
        return new ExecutorHandle();
    MATX_CATCH(nullptr)
}

extern "C" void matx_executor_destroy(MatxExecutorHandle h) {
    delete to_exec(h);
}

extern "C" int matx_executor_sync(MatxExecutorHandle h) {
    MATX_TRY
        return to_exec(h)->sync();
    MATX_CATCH(MATX_ERR_CUDA)
}

extern "C" void* matx_executor_stream(MatxExecutorHandle h) {
    return to_exec(h)->stream();
}

// ── Memory helpers ────────────────────────────────────────────────────────────

extern "C" int matx_tensor_prefetch_device(MatxTensorHandle h, MatxExecutorHandle exec) {
    MATX_TRY
        auto* eh = to_exec(exec);
        cudaStream_t stream = eh->kind == ExecutorHandle::Kind::CUDA
                              ? eh->cuda_exec.getStream()
                              : cudaStream_t{};
        to_base(h)->prefetch(stream);
        return MATX_OK;
    MATX_CATCH(MATX_ERR_CUDA)
}

extern "C" int matx_tensor_copy(MatxTensorHandle dst, MatxTensorHandle src, MatxExecutorHandle exec) {
    MATX_TRY
        auto* d = to_base(dst);
        auto* s = to_base(src);
        if (d->dtype != s->dtype)
            throw std::runtime_error("dtype mismatch in tensor copy");
        if (d->total_size() != s->total_size())
            throw std::runtime_error("element count mismatch in tensor copy");
        size_t bytes = static_cast<size_t>(d->total_size()) * dtype_sizeof(d->dtype);
        auto* eh = to_exec(exec);
        cudaStream_t stream = eh->kind == ExecutorHandle::Kind::CUDA
                              ? eh->cuda_exec.getStream()
                              : cudaStream_t{};
        cudaMemcpyAsync(d->data_ptr(), s->data_ptr(), bytes, cudaMemcpyDefault, stream);
        return MATX_OK;
    MATX_CATCH(MATX_ERR_CUDA)
}

// ── View operations ───────────────────────────────────────────────────────────

extern "C" MatxTensorHandle matx_tensor_flatten(MatxTensorHandle h) {
    MATX_TRY
        auto* base = to_base(h);
        int64_t total = base->total_size();
        return create_view(base->dtype, 1, &total, base->data_ptr());
    MATX_CATCH(nullptr)
}

extern "C" MatxTensorHandle matx_tensor_reshape(
        MatxTensorHandle h, int new_rank, const int64_t* new_shape) {
    MATX_TRY
        auto* base = to_base(h);
        // Validate total size preserved
        int64_t total = 1;
        for (int i = 0; i < new_rank; ++i) total *= new_shape[i];
        if (total != base->total_size())
            throw std::runtime_error("reshape: element count must be preserved");
        return create_view(base->dtype, new_rank, new_shape, base->data_ptr());
    MATX_CATCH(nullptr)
}

// Permute and slice require type-aware dispatch – implemented in matx_elementwise.cu
// using the dispatch helpers to call tensor_t::Permute / tensor_t::Slice.

extern "C" MatxTensorHandle matx_tensor_permute(MatxTensorHandle h, const int32_t* axes) {
    MATX_TRY
        auto* base = to_base(h);
        MatxTensorBase* result = nullptr;
        visit_tensor(base, [&]<typename T, int R>(matx::tensor_t<T, R>& t) {
            int32_t ax[R];
            for (int i = 0; i < R; ++i) ax[i] = axes[i];
            auto perm = t.Permute(ax);          // returns tensor_t<T,R> view
            result = new MatxTypedTensor<T, R>(std::move(perm));
        });
        return result;
    MATX_CATCH(nullptr)
}

extern "C" MatxTensorHandle matx_tensor_slice(
        MatxTensorHandle h,
        const int64_t* starts, const int64_t* ends, const int64_t* strides) {
    MATX_TRY
        auto* base = to_base(h);
        // Convert sentinel values: INT64_MIN → matxKeepDim, INT64_MAX → matxEnd
        MatxTensorBase* result = nullptr;
        visit_tensor(base, [&]<typename T, int R>(matx::tensor_t<T, R>& t) {
            matx::index_t s[R], e[R], st[R];
            for (int i = 0; i < R; ++i) {
                // C# passes INT64_MIN for matxKeepDim and INT64_MAX for matxEnd
                s[i]  = (starts[i]  == INT64_MIN) ? matxKeepDim
                       :(starts[i]  == INT64_MAX) ? matxEnd
                       : static_cast<matx::index_t>(starts[i]);
                e[i]  = (ends[i]    == INT64_MIN) ? matxKeepDim
                       :(ends[i]    == INT64_MAX) ? matxEnd
                       : static_cast<matx::index_t>(ends[i]);
                st[i] = strides ? static_cast<matx::index_t>(strides[i]) : 1;
            }
            auto sliced = t.Slice(s, e, st);
            result = new MatxTypedTensor<T, R>(std::move(sliced));
        });
        return result;
    MATX_CATCH(nullptr)
}
