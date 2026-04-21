# MatX C# Wrapper

A high-performance C# wrapper around [NVIDIA MatX](https://github.com/nvidia/matx) — a header-only CUDA tensor library for GPU-accelerated scientific computing.

## Architecture

```
MatX C# wrapper (MatX.dll)
        │  P/Invoke (LibraryImport, source-generated)
        ▼
matxnative.so / matxnative.dll   ← compiled with nvcc (C++23 + CUDA)
        │  includes
        ▼
NVIDIA MatX  ←  c:\OpenSource\MatX\include\matx.h  (header-only)
```

The bridge is a **pure C API** (`extern "C"`) CUDA shared library. This gives true cross-platform P/Invoke support: the same C# binary runs on Linux, Windows, and macOS (host executor only on macOS).

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| CUDA Toolkit | 11.8 |
| nvcc | matches toolkit |
| CMake | 3.25 |
| .NET SDK | 9.0 |

## Building the native library

```bash
# from the repo root
cd matx-dotnet/native
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Copy output next to the C# project:
#   Linux:   build/libmatxnative.so  →  ../runtimes/linux-x64/native/
#   Windows: build/matxnative.dll    →  ../runtimes/win-x64/native/
```

Override GPU architectures:
```bash
cmake -B build -DMATX_CUDA_ARCH="80;86;90"
```

## Building the C# wrapper

```bash
cd matx-dotnet
dotnet build -c Release
```

## Running tests

```bash
cd matx-dotnet
dotnet test tests/MatX.Tests.csproj -c Release --logger "console;verbosity=normal"
```

GPU tests auto-skip when no CUDA device is present. Host-memory tests always run.

Collect coverage:

```bash
dotnet test tests/MatX.Tests.csproj \
  --collect:"XPlat Code Coverage" \
  --results-directory ./coverage
```

## Quick start

```csharp
using MatX;

// 1. Create a CUDA executor (queue all work onto the default stream)
using var exec = Executor.CreateCuda();

// 2. Allocate tensors in managed (unified) memory
using var a = Tensor.CreateFloat32([1024, 1024]);
using var b = Tensor.CreateFloat32([1024, 1024]);
using var c = Tensor.CreateFloat32([1024, 1024]);

// 3. Fill with data — all operations are asynchronous
Ops.RandomUniform(exec, a);
Ops.RandomUniform(exec, b);

// 4. Chain multiple operations (all queued without intermediate syncs)
using var tmp = Tensor.CreateFloat32([1024, 1024]);
Ops.Add(exec, tmp, a, b);         // tmp = a + b
Ops.Mul(exec, tmp, tmp, 2.0);     // tmp *= 2
Ops.Sqrt(exec, c, tmp);           // c  = sqrt(tmp)

// 5. Synchronize once at the end
exec.Synchronize();

// 6. Read back results
float[] result = c.ToArray<float>();
```

## Chaining calculations efficiently

**Key insight:** every `Ops.*` call just enqueues a kernel on the CUDA stream — no CPU–GPU synchronisation happens between calls. Chain as many operations as needed before calling `Synchronize()`:

```csharp
// Black-Scholes style pipeline — zero intermediate syncs
Ops.Log    (exec, d1, S_over_K);   // d1  = log(S/K)
Ops.Add    (exec, d1, d1, rT);     // d1 += (r+σ²/2)T
Ops.Div    (exec, d1, d1, sigmaT); // d1 /= σ√T
Ops.Sub    (exec, d2, d1, sigmaT); // d2  = d1 − σ√T
Ops.NormCdf(exec, nd1, d1);        // N(d1)
Ops.NormCdf(exec, nd2, d2);        // N(d2)
Ops.Mul    (exec, call, S, nd1);
Ops.Mul    (exec, tmp,  K, nd2);
Ops.Sub    (exec, call, call, tmp);
exec.Synchronize();                // ← single sync for entire pipeline
```

GPU throughput is bottlenecked by kernel launch overhead only when operations are tiny; for large tensors the pipeline is compute-bound and nearly free to extend.

For **view operations** (reshape, permute, slice, flatten) no data is copied — they return a new `Tensor` sharing the same device buffer:

```csharp
using var flat    = matrix.Flatten();          // zero-copy 1-D view
using var transT  = matrix.Permute(1, 0);      // transposed view
using var submat  = matrix.Slice([1,1],[4,4]); // sub-matrix view
```

## Memory spaces

| `MemorySpace` | Description |
|---------------|-------------|
| `Managed` (default) | CUDA unified / managed memory — accessible from both CPU and GPU; use `PrefetchDevice` before GPU work |
| `Device` | GPU-only device memory — fastest for GPU-only pipelines |
| `Host` | Pinned host memory — fast CPU↔GPU transfers |

## Executors

```csharp
using var cudaExec = Executor.CreateCuda();          // default stream
using var stream   = Executor.CreateCuda(myStream);  // custom cudaStream_t
using var hostExec = Executor.CreateHost();          // CPU (no GPU required)
```

## Project layout

```
matx-dotnet/
├── MatX.csproj            # Library project (.NET 9)
├── src/
│   ├── Tensor.cs          # Tensor wrapper + view API
│   ├── Executor.cs        # Executor wrapper
│   ├── Ops.cs             # All operations (generators, math, FFT, linalg, signal)
│   ├── Enums.cs           # DType, MemorySpace, FftNorm, …
│   ├── MatxException.cs
│   ├── NativeHelpers.cs
│   └── Native/
│       └── NativeLib.cs   # All P/Invoke declarations
├── native/
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── matx_api.h     # Public C API header
│   └── src/
│       ├── matx_core.cu
│       ├── matx_elementwise.cu
│       ├── matx_generators.cu
│       ├── matx_fft.cu
│       ├── matx_linalg.cu
│       └── matx_signal.cu
├── tests/
│   ├── MatX.Tests.csproj
│   ├── TensorTests.cs     # Tensor lifecycle, metadata, views, data access
│   ├── ExecutorTests.cs   # Executor lifecycle (host + CUDA skip)
│   ├── OpsHostTests.cs    # All generators, unary, binary, reduction ops (CPU)
│   └── OpsCudaTests.cs    # FFT, linalg, signal processing (GPU, auto-skip)
└── examples/
    ├── MatX.Examples.csproj
    ├── FftConvolutionExample.cs
    ├── SvdPowerExample.cs
    ├── BlackScholesExample.cs
    ├── SpectrogramExample.cs
    └── DecompositionsExample.cs
```

## License

Same as NVIDIA MatX — Apache 2.0. See [LICENSE](../LICENSE).
