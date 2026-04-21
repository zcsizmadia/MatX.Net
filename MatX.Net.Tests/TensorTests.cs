// TensorTests.cs – Tests for Tensor creation, metadata, data access, and view ops.
// These tests do NOT require a GPU: they create host-memory tensors via
// MemorySpace.Host and use Executor.CreateHost().
using MatX;
using Xunit;

namespace MatX.Net.Tests;

public sealed class TensorCreateTests
{
    [Fact] public void CreateFloat32_CorrectDtype()
    {
        using var t = Tensor.CreateFloat32([4]);
        Assert.Equal(DType.Float32, t.DataType);
    }

    [Fact] public void CreateFloat64_CorrectDtype()
    {
        using var t = Tensor.CreateFloat64([4]);
        Assert.Equal(DType.Float64, t.DataType);
    }

    [Fact] public void CreateInt32_CorrectDtype()
    {
        using var t = Tensor.CreateInt32([4]);
        Assert.Equal(DType.Int32, t.DataType);
    }

    [Fact] public void CreateInt64_CorrectDtype()
    {
        using var t = Tensor.CreateInt64([4]);
        Assert.Equal(DType.Int64, t.DataType);
    }

    [Fact] public void CreateComplex64_CorrectDtype()
    {
        using var t = Tensor.CreateComplex64([4]);
        Assert.Equal(DType.Complex64, t.DataType);
    }

    [Fact] public void CreateComplex128_CorrectDtype()
    {
        using var t = Tensor.CreateComplex128([4]);
        Assert.Equal(DType.Complex128, t.DataType);
    }

    [Theory]
    [InlineData(new long[] { 8 },       1, 8)]
    [InlineData(new long[] { 4, 4 },    2, 16)]
    [InlineData(new long[] { 2, 3, 4 }, 3, 24)]
    public void Create_CorrectRankAndTotalSize(long[] shape, int expectedRank, long expectedTotal)
    {
        using var t = Tensor.CreateFloat32(shape);
        Assert.Equal(expectedRank, t.Rank);
        Assert.Equal(expectedTotal, t.TotalSize);
    }

    [Fact] public void Shape_ReturnsCorrectArray()
    {
        long[] shape = [3, 5, 7];
        using var t = Tensor.CreateFloat32(shape);
        Assert.Equal(shape, t.Shape);
    }

    [Fact] public void Size_PerDimension()
    {
        using var t = Tensor.CreateFloat32([2, 6]);
        Assert.Equal(2L, t.Size(0));
        Assert.Equal(6L, t.Size(1));
    }

    [Fact] public void DataPointer_NotZero()
    {
        using var t = Tensor.CreateFloat32([8]);
        Assert.NotEqual(IntPtr.Zero, t.DataPointer);
    }

    [Fact] public void Name_CanBeSet()
    {
        using var t = Tensor.CreateFloat32([4]);
        t.Name = "my_tensor"; // must not throw
    }

    [Fact] public void NullShape_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => Tensor.CreateFloat32(null!));
    }

    [Fact] public void Dispose_TwiceDoesNotThrow()
    {
        var t = Tensor.CreateFloat32([4]);
        t.Dispose();
        t.Dispose(); // must be idempotent
    }

    [Fact] public void AccessAfterDispose_Throws()
    {
        var t = Tensor.CreateFloat32([4]);
        t.Dispose();
        Assert.Throws<ObjectDisposedException>(() => t.Rank);
    }
}

public sealed class TensorDataAccessTests
{
    [Fact] public void CopyFromAndToArray_RoundTrip()
    {
        float[] data = [1f, 2f, 3f, 4f, 5f, 6f];
        using var t = Tensor.CreateFloat32([6], MemorySpace.Host);
        t.CopyFrom<float>(data);
        float[] result = t.ToArray<float>();
        Assert.Equal(data, result);
    }

    [Fact] public void CopyFrom_WrongLength_Throws()
    {
        using var t = Tensor.CreateFloat32([4], MemorySpace.Host);
        Assert.Throws<ArgumentException>(() => t.CopyFrom<float>([1f, 2f]));
    }

    [Fact] public void CopyFrom_Int32_RoundTrip()
    {
        int[] data = [10, 20, 30, 40];
        using var t = Tensor.CreateInt32([4], MemorySpace.Host);
        t.CopyFrom<int>(data);
        Assert.Equal(data, t.ToArray<int>());
    }

    [Fact] public void CopyFrom_Double_RoundTrip()
    {
        double[] data = [Math.PI, Math.E, 1.0, 0.0];
        using var t = Tensor.CreateFloat64([4], MemorySpace.Host);
        t.CopyFrom<double>(data);
        double[] r = t.ToArray<double>();
        Assert.Equal(data, r);
    }

    [Fact] public void FromPointer_ViewSharesData()
    {
        float[] data = [10f, 20f, 30f, 40f];
        unsafe
        {
            fixed (float* p = data)
            {
                using var t = Tensor.FromPointer(DType.Float32, [4], (IntPtr)p);
                Assert.Equal(DType.Float32, t.DataType);
                Assert.Equal(4L, t.TotalSize);
            }
        }
    }
}

public sealed class TensorViewTests
{
    [Fact] public void Flatten_ReducesToOneDim()
    {
        using var t  = Tensor.CreateFloat32([4, 4]);
        using var fl = t.Flatten();
        Assert.Equal(1, fl.Rank);
        Assert.Equal(16L, fl.TotalSize);
    }

    [Fact] public void Reshape_CorrectShape()
    {
        using var t  = Tensor.CreateFloat32([8]);
        using var r  = t.Reshape(2, 4);
        Assert.Equal(2, r.Rank);
        Assert.Equal(2L, r.Size(0));
        Assert.Equal(4L, r.Size(1));
    }

    [Fact] public void Permute_SwapsAxes()
    {
        using var t  = Tensor.CreateFloat32([3, 5]);
        using var tp = t.Permute(1, 0);
        Assert.Equal(5L, tp.Size(0));
        Assert.Equal(3L, tp.Size(1));
    }

    [Fact] public void Permute_3D()
    {
        using var t  = Tensor.CreateFloat32([2, 3, 4]);
        using var tp = t.Permute(2, 0, 1);
        Assert.Equal(4L, tp.Size(0));
        Assert.Equal(2L, tp.Size(1));
        Assert.Equal(3L, tp.Size(2));
    }

    [Fact] public void Slice_ReducesSize()
    {
        using var t  = Tensor.CreateFloat32([10]);
        // Slice [2..6)
        using var sl = t.Slice(starts: [2], ends: [6]);
        Assert.Equal(4L, sl.Size(0));
    }
}
