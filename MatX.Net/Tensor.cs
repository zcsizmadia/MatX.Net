// Tensor.cs – Managed wrapper around a MatX tensor.
// Zero-overhead: the C# object is just a wrapper holding one IntPtr.
using System.Numerics;
using System.Runtime.InteropServices;
using MatX.Net.Native;

namespace MatX.Net;

/// <summary>
/// A multi-dimensional array living in CUDA or host memory.
/// Wraps a native <c>MatxTensorHandle</c>.
/// </summary>
/// <remarks>
/// All mutation methods execute operations through an <see cref="Executor"/>.
/// To avoid repeated sync overhead, batch operations and call
/// <see cref="Executor.Synchronize"/> once at the end.
/// </remarks>
public sealed class Tensor : IDisposable
{
    private IntPtr _handle;
    private bool   _disposed;

    // ────────────────────────────────────────────────────────────────────────
    //  Constructors / Factories
    // ────────────────────────────────────────────────────────────────────────

    internal Tensor(IntPtr handle) => _handle = handle;

    /// <summary>Allocate a new owning tensor.</summary>
    public static Tensor Create(DType dtype, long[] shape,
        MemorySpace memory = MemorySpace.Managed)
    {
        ArgumentNullException.ThrowIfNull(shape);
        unsafe
        {
            fixed (long* s = shape)
                return new(NativeLib.CheckHandle(
                    NativeLib.TensorCreatePtr(dtype, shape.Length, s, memory)));
        }
    }

    /// <summary>Create a float32 tensor.</summary>
    public static Tensor CreateFloat32(long[] shape, MemorySpace memory = MemorySpace.Managed)
        => Create(DType.Float32, shape, memory);

    /// <summary>Create a float64 tensor.</summary>
    public static Tensor CreateFloat64(long[] shape, MemorySpace memory = MemorySpace.Managed)
        => Create(DType.Float64, shape, memory);

    /// <summary>Create a complex64 (float real + float imaginary) tensor.</summary>
    public static Tensor CreateComplex64(long[] shape, MemorySpace memory = MemorySpace.Managed)
        => Create(DType.Complex64, shape, memory);

    /// <summary>Create a complex128 (double real + double imaginary) tensor.</summary>
    public static Tensor CreateComplex128(long[] shape, MemorySpace memory = MemorySpace.Managed)
        => Create(DType.Complex128, shape, memory);

    /// <summary>Create an int32 tensor.</summary>
    public static Tensor CreateInt32(long[] shape, MemorySpace memory = MemorySpace.Managed)
        => Create(DType.Int32, shape, memory);

    /// <summary>Create an int64 tensor.</summary>
    public static Tensor CreateInt64(long[] shape, MemorySpace memory = MemorySpace.Managed)
        => Create(DType.Int64, shape, memory);

    /// <summary>
    /// Wrap an existing raw pointer as a non-owning view.
    /// The caller is responsible for keeping the pointer alive.
    /// </summary>
    public static Tensor FromPointer(DType dtype, long[] shape, IntPtr data)
    {
        ArgumentNullException.ThrowIfNull(shape);
        unsafe
        {
            fixed (long* s = shape)
                return new(NativeLib.CheckHandle(
                    NativeLib.TensorFromPtr(dtype, shape.Length, s, (void*)data)));
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Metadata
    // ────────────────────────────────────────────────────────────────────────

    internal IntPtr Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    public int    Rank       => NativeLib.TensorRank(Handle);
    public DType  DataType   => NativeLib.TensorDtype(Handle);
    public long   TotalSize  => NativeLib.TensorTotalSize(Handle);
    public IntPtr DataPointer => NativeLib.TensorData(Handle);

    /// <summary>Size of dimension <paramref name="dim"/>.</summary>
    public long Size(int dim) => NativeLib.TensorSize(Handle, dim);

    /// <summary>Stride (in elements) of dimension <paramref name="dim"/>.</summary>
    public long Stride(int dim) => NativeLib.TensorStride(Handle, dim);

    /// <summary>Shape array (allocates on each call; cache when needed).</summary>
    public long[] Shape
    {
        get
        {
            var r = Rank;
            var s = new long[r];
            for (int i = 0; i < r; i++) s[i] = Size(i);
            return s;
        }
    }

    /// <summary>Set a debug name visible in MatX print output.</summary>
    public string? Name
    {
        set => NativeLib.TensorSetName(Handle, value ?? "");
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Data access helpers
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Copy the tensor's elements into a managed array.
    /// Requires managed or host memory (cudaMemcpyDefault).
    /// </summary>
    public T[] ToArray<T>() where T : unmanaged
    {
        long total = TotalSize;
        var  result = new T[total];
        unsafe
        {
            fixed (T* dst = result)
            {
                Buffer.MemoryCopy(
                    (void*)DataPointer,
                    dst,
                    total * sizeof(T),
                    total * sizeof(T));
            }
        }
        return result;
    }

    /// <summary>
    /// Copy values from a managed span into the tensor's buffer.
    /// Tensor must be in managed or host memory.
    /// </summary>
    public void CopyFrom<T>(ReadOnlySpan<T> data) where T : unmanaged
    {
        if (data.Length != TotalSize)
            throw new ArgumentException("data length must equal TotalSize");
        unsafe
        {
            fixed (T* src = data)
                Buffer.MemoryCopy(src, (void*)DataPointer,
                    data.Length * sizeof(T),
                    data.Length * sizeof(T));
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  View operations  (return a new Tensor sharing the same buffer)
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>Reshape to a new shape preserving total element count.</summary>
    public Tensor Reshape(params long[] newShape)
    {
        unsafe
        {
            fixed (long* s = newShape)
                return new(NativeLib.CheckHandle(
                    NativeLib.TensorReshape(Handle, newShape.Length, s)));
        }
    }

    /// <summary>Transpose / reorder axes. <paramref name="axes"/> is a permutation of [0..rank-1].</summary>
    public Tensor Permute(params int[] axes)
    {
        unsafe
        {
            fixed (int* a = axes)
                return new(NativeLib.CheckHandle(
                    NativeLib.TensorPermute(Handle, a)));
        }
    }

    /// <summary>Flatten to a 1-D view.</summary>
    public Tensor Flatten()
        => new(NativeLib.CheckHandle(NativeLib.TensorFlatten(Handle)));

    /// <summary>
    /// Sub-tensor slice view.
    /// Use <c>long.MinValue</c> for <c>matxKeepDim</c> and
    /// <c>long.MaxValue</c> for <c>matxEnd</c>.
    /// </summary>
    public Tensor Slice(long[] starts, long[] ends, long[]? strides = null)
    {
        int r = Rank;
        if (starts.Length != r || ends.Length != r)
            throw new ArgumentException("starts and ends must have length == Rank");
        if (strides is not null && strides.Length != r)
            throw new ArgumentException("strides must have length == Rank");

        unsafe
        {
            fixed (long* s = starts, e = ends)
            {
                if (strides is not null)
                    fixed (long* st = strides)
                        return new(NativeLib.CheckHandle(
                            NativeLib.TensorSlice(Handle, s, e, st)));
                else
                    return new(NativeLib.CheckHandle(
                        NativeLib.TensorSlice(Handle, s, e, null)));
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Memory helpers
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Hint the CUDA driver to migrate this tensor to the GPU before the next kernel.
    /// Only meaningful for managed-memory tensors.
    /// </summary>
    public void PrefetchDevice(Executor exec)
        => NativeLib.Check(NativeLib.TensorPrefetchDevice(Handle, exec.Handle));

    /// <summary>Async GPU→GPU or GPU→host copy into this tensor.</summary>
    public void CopyFrom(Tensor src, Executor exec)
        => NativeLib.Check(NativeLib.TensorCopy(Handle, src.Handle, exec.Handle));

    // ────────────────────────────────────────────────────────────────────────
    //  Print
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>Print the tensor to stdout via MatX's built-in formatter.</summary>
    public void Print() => NativeLib.TensorPrint(Handle);

    // ────────────────────────────────────────────────────────────────────────
    //  Disposal
    // ────────────────────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        NativeLib.TensorDestroy(_handle);
        _handle = IntPtr.Zero;
    }

    public override string ToString()
        => $"Tensor<{DataType}>[{string.Join(", ", Shape)}]";
}
