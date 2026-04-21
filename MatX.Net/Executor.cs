// Executor.cs – Managed wrapper around a MatX executor.
using MatX.Net.Native;

namespace MatX.Net;

/// <summary>
/// An executor determines where and how tensor operations run.
/// Use <see cref="CreateCuda()"/> for GPU execution or <see cref="CreateHost"/> for CPU.
/// </summary>
public sealed class Executor : IDisposable
{
    private IntPtr _handle;
    private bool   _disposed;

    // ── Factories ─────────────────────────────────────────────────────────────

    /// <summary>Create a CUDA executor using the default stream (0).</summary>
    public static Executor CreateCuda() => new(NativeLib.CheckHandle(
        NativeLib.ExecutorCuda(IntPtr.Zero)));

    /// <summary>Create a CUDA executor on a specific stream (raw cudaStream_t pointer).</summary>
    public static Executor CreateCuda(IntPtr stream) => new(NativeLib.CheckHandle(
        NativeLib.ExecutorCuda(stream)));

    /// <summary>Create a CPU (host) executor.</summary>
    public static Executor CreateHost() => new(NativeLib.CheckHandle(
        NativeLib.ExecutorHost()));

    private Executor(IntPtr handle) => _handle = handle;

    // ── Properties ────────────────────────────────────────────────────────────

    internal IntPtr Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    /// <summary>Returns the underlying cudaStream_t as a raw pointer (null for host executors).</summary>
    public IntPtr Stream => NativeLib.ExecutorStream(Handle);

    // ── Methods ───────────────────────────────────────────────────────────────

    /// <summary>Block until all operations submitted to this executor have completed.</summary>
    public void Synchronize() => NativeLib.Check(NativeLib.ExecutorSync(Handle));

    // ── Disposal ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Releases the unmanaged resources used by the executor.
    /// </summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        NativeLib.ExecutorDestroy(_handle);
        _handle = IntPtr.Zero;
    }
}
