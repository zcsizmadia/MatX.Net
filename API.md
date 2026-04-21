# MatX C# API Reference

## Namespace `MatX`

---

### `Tensor`

Represents a multi-dimensional array in CUDA or host memory. Wraps a native `MatxTensorHandle`. Implements `IDisposable`.

#### Factory methods

| Method | Description |
|--------|-------------|
| `Tensor.Create(DType dtype, long[] shape, MemorySpace memory = Managed)` | Allocate a new owning tensor. |
| `Tensor.CreateFloat32(long[] shape, MemorySpace memory = Managed)` | Shorthand for `DType.Float32`. |
| `Tensor.CreateFloat64(long[] shape, MemorySpace memory = Managed)` | Shorthand for `DType.Float64`. |
| `Tensor.CreateComplex64(long[] shape, MemorySpace memory = Managed)` | Complex (float32 real + float32 imag). |
| `Tensor.CreateComplex128(long[] shape, MemorySpace memory = Managed)` | Complex (float64 real + float64 imag). |
| `Tensor.CreateInt32(long[] shape, MemorySpace memory = Managed)` | 32-bit integer. |
| `Tensor.CreateInt64(long[] shape, MemorySpace memory = Managed)` | 64-bit integer. |
| `Tensor.FromPointer(DType dtype, long[] shape, IntPtr data)` | Non-owning view over existing memory. Caller manages lifetime. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `Rank` | `int` | Number of dimensions. |
| `DataType` | `DType` | Element type. |
| `TotalSize` | `long` | Product of all dimension sizes. |
| `DataPointer` | `IntPtr` | Raw pointer to the first element. |
| `Shape` | `long[]` | Dimension sizes array (allocates on each call). |
| `Name` | `string?` (set only) | Debug label shown in MatX print output. |

#### Per-dimension accessors

| Method | Description |
|--------|-------------|
| `long Size(int dim)` | Size of dimension `dim`. |
| `long Stride(int dim)` | Stride (in elements) of dimension `dim`. |

#### Data transfer

| Method | Description |
|--------|-------------|
| `T[] ToArray<T>()` | Copy tensor data to a managed array. Tensor must be in managed or host memory. |
| `void CopyFrom<T>(ReadOnlySpan<T> data)` | Copy a managed span into the tensor buffer. Length must equal `TotalSize`. |

#### View operations (zero-copy)

All view methods return a new `Tensor` that shares the underlying buffer. Dispose the view when done.

| Method | Description |
|--------|-------------|
| `Tensor Flatten()` | 1-D view of all elements. |
| `Tensor Reshape(params long[] newShape)` | New shape with same total element count. |
| `Tensor Permute(params int[] axes)` | Reorder axes. `axes` is a permutation of `[0..rank-1]`. |
| `Tensor Slice(long[] starts, long[] ends, long[]? strides = null)` | Sub-tensor view. Use `long.MinValue` for `matxKeepDim`, `long.MaxValue` for `matxEnd`. |

#### Memory helpers

| Method | Description |
|--------|-------------|
| `void PrefetchDevice(Executor exec)` | Hint CUDA to migrate managed-memory pages to the GPU. |
| `void CopyFrom(Tensor src, Executor exec)` | Async device-to-device (or any-to-any) copy. |
| `void Print()` | Print tensor contents via MatX's built-in pretty-printer. |

---

### `Executor`

Determines where and how tensor operations run. Implements `IDisposable`.

#### Factory methods

| Method | Description |
|--------|-------------|
| `Executor.CreateCuda()` | CUDA executor on the default stream (0). |
| `Executor.CreateCuda(IntPtr stream)` | CUDA executor on a custom `cudaStream_t`. |
| `Executor.CreateHost()` | CPU (host) executor. |

#### Methods and properties

| Member | Description |
|--------|-------------|
| `void Synchronize()` | Block until all queued operations complete. |
| `IntPtr Stream` | Underlying `cudaStream_t` (zero for host executor). |

---

### `Ops` (static class)

All operations are **asynchronous** — they enqueue work on the executor's CUDA stream and return immediately. Call `exec.Synchronize()` once after a batch of operations.

#### Generators — fill an existing tensor

| Method | Description |
|--------|-------------|
| `Zeros(exec, dst)` | Fill with 0. |
| `Ones(exec, dst)` | Fill with 1. |
| `RandomUniform(exec, dst)` | Fill with U[0,1) values. |
| `RandomNormal(exec, dst)` | Fill with N(0,1) values. |
| `Linspace(exec, dst, start, stop)` | 1-D arithmetic sequence from `start` to `stop`. |
| `Range(exec, dst, start, step)` | 1-D arithmetic sequence with given step. |
| `FftFreq(exec, dst, sampleRate = 1.0)` | 1-D FFT sample frequencies. |
| `Hamming(exec, dst)` | Hamming window. |
| `Hanning(exec, dst)` | Hanning (von Hann) window. |
| `Blackman(exec, dst)` | Blackman window. |
| `Bartlett(exec, dst)` | Bartlett window. |

#### Unary element-wise  `Foo(exec, dst, a)`

`Abs`, `Abs2`, `Sqrt`, `Exp`, `Log`, `Log2`, `Log10`, `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Tanh`, `Ceil`, `Floor`, `Round`, `Sign`, `Neg`, `Conj`, `Real`, `Imag`, `NormCdf`, `Erf`, `Erfc`

#### Binary element-wise  `Foo(exec, dst, a, b)`  and scalar variants

| Tensor–tensor | Scalar variant | Description |
|---------------|---------------|-------------|
| `Add(exec, dst, a, b)` | `Add(exec, dst, a, scalar)` | Element-wise addition |
| `Sub(exec, dst, a, b)` | `Sub(exec, dst, a, scalar)` | Element-wise subtraction |
| `Mul(exec, dst, a, b)` | `Mul(exec, dst, a, scalar)` | Element-wise multiplication |
| `Div(exec, dst, a, b)` | `Div(exec, dst, a, scalar)` | Element-wise division |
| `Pow(exec, dst, a, b)` | `Pow(exec, dst, a, scalar)` | Element-wise power |
| `Maximum(exec, dst, a, b)` | — | Element-wise max |
| `Minimum(exec, dst, a, b)` | — | Element-wise min |

#### Reductions

`axis = -1` reduces over all dimensions and writes a scalar to `dst`.

| Method | Description |
|--------|-------------|
| `Sum(exec, dst, src, axis = -1)` | Sum. |
| `Mean(exec, dst, src, axis = -1)` | Mean. |
| `Max(exec, dst, src, axis = -1)` | Maximum value. |
| `Min(exec, dst, src, axis = -1)` | Minimum value. |
| `Prod(exec, dst, src, axis = -1)` | Product. |
| `Var(exec, dst, src, axis = -1)` | Variance. |
| `Std(exec, dst, src, axis = -1)` | Standard deviation. |
| `Any(exec, dst, src)` | True if any element is non-zero. |
| `Norm(exec, dst, src)` | Frobenius / L2 norm. |
| `Trace(exec, dst, src)` | Sum of diagonal (rank-2 only). |
| `CumSum(exec, dst, src, axis = 0)` | Cumulative sum along axis. |
| `Sort(exec, dst, src, dir = Ascending)` | Sorted copy. |

#### FFT

| Method | Description |
|--------|-------------|
| `Fft(exec, dst, src, nfft = 0, norm = Backward)` | 1-D complex FFT. `nfft=0` uses source length. |
| `Ifft(exec, dst, src, nfft = 0, norm = Backward)` | 1-D inverse complex FFT. |
| `Fft2(exec, dst, src, norm = Backward)` | 2-D complex FFT. |
| `Ifft2(exec, dst, src, norm = Backward)` | 2-D inverse complex FFT. |
| `Rfft(exec, dst, src, nfft = 0, norm = Backward)` | Real → complex FFT. Output size is `nfft/2+1`. |
| `Irfft(exec, dst, src, nfft = 0, norm = Backward)` | Complex → real IFFT. |
| `FftShift(exec, dst, src)` | Shift zero-frequency to centre. |
| `IFftShift(exec, dst, src)` | Inverse shift. |
| `Dct(exec, dst, src)` | Discrete cosine transform (type-II). |

#### Linear algebra

| Method | Description |
|--------|-------------|
| `Matmul(exec, dst, a, b)` | GEMM. Supports rank-2 and batched rank-3. |
| `Matvec(exec, dst, a, x)` | Matrix × vector. |
| `Outer(exec, dst, a, b)` | Outer product (rank-1 × rank-1 → rank-2). |
| `Transpose(exec, dst, src)` | Rank-2 transpose. |
| `Hermitian(exec, dst, src)` | Conjugate transpose. |
| `Svd(exec, u, s, vt, a, mode = Reduced)` | Full or reduced SVD. |
| `SvdPi(exec, u, s, vt, a, iterations = 10, k = -1)` | Truncated SVD via power iteration. `k=-1` uses `a.Size(1)`. |
| `Qr(exec, q, r, a)` | QR decomposition. |
| `Lu(exec, l, u, piv, a)` | LU decomposition with pivoting. |
| `Chol(exec, l, a)` | Cholesky decomposition (lower triangular). |
| `Eig(exec, vec, val, a)` | Eigendecomposition (symmetric/Hermitian). |
| `Solve(exec, x, a, b)` | Solve A·x = b. |
| `Inv(exec, dst, a)` | Matrix inverse. |
| `Pinv(exec, dst, a)` | Moore-Penrose pseudo-inverse. |
| `Det(exec, dst, a)` | Determinant (scalar output). |
| `Einsum(exec, dst, subscripts, a, b)` | Einstein summation, e.g. `"ij,jk->ik"`. |

#### Signal processing

| Method | Description |
|--------|-------------|
| `Conv1d(exec, dst, signal, filter, mode = Full)` | 1-D convolution. |
| `Conv2d(exec, dst, signal, filter, mode = Full)` | 2-D convolution. |
| `Corr(exec, dst, a, b)` | 1-D cross-correlation (full mode). |
| `Pwelch(exec, pxx, signal, window, noverlap, nfft)` | Welch PSD estimate. `signal` shape: `(batch, N)`. `pxx` shape: `(batch, nfft/2+1)`. |
| `ResamplePoly(exec, dst, src, up, down, filter = null)` | Polyphase rational-rate resampling. |

---

### Enums

#### `DType`

| Value | C++ type | Bytes |
|-------|----------|-------|
| `Float32` | `float` | 4 |
| `Float64` | `double` | 8 |
| `Complex64` | `cuda::std::complex<float>` | 8 |
| `Complex128` | `cuda::std::complex<double>` | 16 |
| `Int32` | `int32_t` | 4 |
| `Int64` | `int64_t` | 8 |

#### `MemorySpace`

| Value | Description |
|-------|-------------|
| `Managed` | CUDA unified memory (default). |
| `Device` | GPU device memory. |
| `Host` | Pinned host memory. |

#### `FftNorm`

| Value | Normalisation |
|-------|--------------|
| `Backward` | No normalisation on forward; divide by N on inverse. |
| `Ortho` | Divide by √N on both forward and inverse. |
| `Forward` | Divide by N on forward; no normalisation on inverse. |

#### `ConvMode`

| Value | Output size |
|-------|------------|
| `Full` | `M + N - 1` |
| `Same` | `max(M, N)` |
| `Valid` | `max(M, N) - min(M, N) + 1` |

#### `SortDir`

`Ascending`, `Descending`

#### `SvdMode`

`Reduced`, `Full`, `None`

---

### `MatxException`

```csharp
public sealed class MatxException : Exception
{
    public int ErrorCode { get; }
}
```

Thrown when a native function returns a non-zero error code. `Message` contains the last error string retrieved from the native thread-local error buffer.

---

## Error codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Invalid argument |
| 2 | CUDA runtime error |
| 3 | cuBLAS / cuSolver error |
| 4 | Unsupported dtype or rank |
| 5 | Unknown / internal error |
