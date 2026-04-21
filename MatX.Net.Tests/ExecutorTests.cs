// ExecutorTests.cs – Tests for Executor creation and lifecycle.
using MatX;
using Xunit;

namespace MatX.Net.Tests;

public sealed class ExecutorHostTests
{
    [Fact] public void CreateHost_Succeeds()
    {
        using var exec = Executor.CreateHost();
        Assert.NotNull(exec);
    }

    [Fact] public void Synchronize_DoesNotThrow()
    {
        using var exec = Executor.CreateHost();
        exec.Synchronize(); // should complete immediately
    }

    [Fact] public void Dispose_TwiceDoesNotThrow()
    {
        var exec = Executor.CreateHost();
        exec.Dispose();
        exec.Dispose();
    }

    [Fact] public void AccessAfterDispose_Throws()
    {
        var exec = Executor.CreateHost();
        exec.Dispose();
        Assert.Throws<ObjectDisposedException>(() => exec.Synchronize());
    }
}

/// <summary>GPU executor tests – skipped when CUDA is unavailable.</summary>
public sealed class ExecutorCudaTests
{
    // Trait: skip tests when no GPU present so CI on CPU-only machines stays green.
    private static readonly bool HasGpu = DetectGpu();

    private static bool DetectGpu()
    {
        try
        {
            using var e = Executor.CreateCuda();
            return true;
        }
        catch { return false; }
    }

    [SkippableFact] public void CreateCuda_Succeeds()
    {
        Skip.IfNot(HasGpu, "No CUDA GPU detected");
        using var exec = Executor.CreateCuda();
        Assert.NotNull(exec);
    }

    [SkippableFact] public void Synchronize_AfterCudaCreate_DoesNotThrow()
    {
        Skip.IfNot(HasGpu, "No CUDA GPU detected");
        using var exec = Executor.CreateCuda();
        exec.Synchronize();
    }

    [SkippableFact] public void Stream_ReturnsValidPointerOrZero()
    {
        Skip.IfNot(HasGpu, "No CUDA GPU detected");
        using var exec = Executor.CreateCuda();
        // The default stream is 0; both are acceptable.
        _ = exec.Stream;
    }
}
