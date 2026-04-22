// executor_handle.h  –  Type-erased executor wrapper.
#pragma once

#include <matx.h>
#include <stdexcept>
#include <variant>
#include "matx_api.h"

struct ExecutorHandle {
    enum class Kind { CUDA, HOST } kind;

    matx::cudaExecutor  cuda_exec{};
    matx::SingleThreadedHostExecutor  host_exec{};

    explicit ExecutorHandle(cudaStream_t stream)
        : kind(Kind::CUDA), cuda_exec(stream) {}

    explicit ExecutorHandle()
        : kind(Kind::HOST) {}

    /// Run a callable with the concrete executor type.
    /// Callable must be a generic lambda accepting (auto& exec).
    template<typename Fn>
    void run(Fn&& fn) {
        if (kind == Kind::CUDA) fn(cuda_exec);
        else                    fn(host_exec);
    }

    int sync() {
        if (kind == Kind::CUDA) {
            return static_cast<int>(cudaStreamSynchronize(cuda_exec.getStream()));
        }
        return 0;
    }

    void* stream() const noexcept {
        if (kind == Kind::CUDA) return reinterpret_cast<void*>(cuda_exec.getStream());
        return nullptr;
    }
};
