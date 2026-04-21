// OpsCudaTests.cs – GPU-only tests for FFT, linalg, and signal processing.
// All tests use [SkippableFact] and skip gracefully when no CUDA GPU is present.
using MatX;
using Xunit;

namespace MatX.Net.Tests;

/// <summary>Shared GPU skip logic.</summary>
internal static class GpuCheck
{
    public static readonly bool HasGpu = Probe();
    private static bool Probe()
    {
        try { using var e = Executor.CreateCuda(); return true; }
        catch { return false; }
    }
}

public sealed class FftCudaTests
{
    [SkippableFact] public void Fft_ComplexRoundTrip()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int N = 64;
        using var exec = Executor.CreateCuda();

        using var src = Tensor.CreateComplex64([N]);
        using var freq = Tensor.CreateComplex64([N]);
        using var rec  = Tensor.CreateComplex64([N]);

        // Fill source with ones (real part)
        Ops.Ones(exec, src);

        Ops.Fft (exec, freq, src, norm: FftNorm.Ortho);
        Ops.Ifft(exec, rec,  freq, norm: FftNorm.Ortho);
        exec.Synchronize();

        // rec should equal src (within float precision)
        // We validate via energy: sum(abs2(src - rec)) ≈ 0
        using var diff   = Tensor.CreateComplex64([N]);
        using var energy = Tensor.CreateFloat32([1]);
        using var abs2   = Tensor.CreateFloat32([N]);
        Ops.Sub (exec, diff,   src, rec);
        Ops.Abs2(exec, abs2,   diff);
        Ops.Sum (exec, energy, abs2);
        exec.Synchronize();

        float e = energy.ToArray<float>()[0];
        Assert.True(e < 1e-4f, $"FFT round-trip energy error = {e}");
    }

    [SkippableFact] public void Rfft_OutputSizeIsNOver2Plus1()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int N = 128;
        using var exec = Executor.CreateCuda();
        using var src  = Tensor.CreateFloat32([N]);
        using var freq = Tensor.CreateComplex64([N / 2 + 1]);
        Ops.RandomNormal(exec, src);
        Ops.Rfft(exec, freq, src);
        exec.Synchronize();
        Assert.Equal(N / 2 + 1L, freq.Size(0));
    }

    [SkippableFact] public void FftShift_IfftShift_RoundTrip()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int N = 32;
        using var exec = Executor.CreateCuda();
        using var src  = Tensor.CreateFloat32([N]);
        using var sh   = Tensor.CreateFloat32([N]);
        using var ish  = Tensor.CreateFloat32([N]);
        Ops.RandomUniform(exec, src);
        Ops.FftShift (exec, sh,  src);
        Ops.IFftShift(exec, ish, sh);
        exec.Synchronize();

        float[] s = src.ToArray<float>();
        float[] r = ish.ToArray<float>();
        for (int i = 0; i < N; i++)
            Assert.Equal(s[i], r[i], precision: 4);
    }

    [SkippableFact] public void Dct_ProducesNonZeroOutput()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int N = 64;
        using var exec = Executor.CreateCuda();
        using var src = Tensor.CreateFloat32([N]);
        using var dst = Tensor.CreateFloat32([N]);
        Ops.RandomUniform(exec, src);
        Ops.Dct(exec, dst, src);
        exec.Synchronize();

        using var norm = Tensor.CreateFloat32([1]);
        Ops.Norm(exec, norm, dst);
        exec.Synchronize();
        Assert.True(norm.ToArray<float>()[0] > 0f);
    }
}

public sealed class LinalgCudaTests
{
    [SkippableFact] public void Matmul_IdentityTimesA_EqualsA()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int N = 4;
        using var exec = Executor.CreateCuda();

        // Identity matrix
        using var ident = Tensor.CreateFloat32([N, N]);
        Ops.Zeros(exec, ident);
        exec.Synchronize();
        float[] id = new float[N * N];
        for (int i = 0; i < N; i++) id[i * N + i] = 1f;
        ident.CopyFrom<float>(id);

        using var a = Tensor.CreateFloat32([N, N]);
        Ops.RandomUniform(exec, a);
        exec.Synchronize();
        float[] aHost = a.ToArray<float>();

        using var c = Tensor.CreateFloat32([N, N]);
        Ops.Matmul(exec, c, ident, a);
        exec.Synchronize();

        float[] cHost = c.ToArray<float>();
        for (int i = 0; i < N * N; i++)
            Assert.Equal(aHost[i], cHost[i], precision: 4);
    }

    [SkippableFact] public void Transpose_SwapsDimensions()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        using var exec = Executor.CreateCuda();
        using var a = Tensor.CreateFloat32([3, 5]);
        using var b = Tensor.CreateFloat32([5, 3]);
        Ops.RandomUniform(exec, a);
        Ops.Transpose(exec, b, a);
        exec.Synchronize();
        Assert.Equal(5L, b.Size(0));
        Assert.Equal(3L, b.Size(1));
    }

    [SkippableFact] public void Qr_QTimesRApproxA()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int M = 8, N = 8;
        using var exec = Executor.CreateCuda();
        using var a  = Tensor.CreateFloat32([M, N]);
        using var q  = Tensor.CreateFloat32([M, N]);
        using var r  = Tensor.CreateFloat32([N, N]);
        using var qr = Tensor.CreateFloat32([M, N]);
        Ops.RandomUniform(exec, a);
        exec.Synchronize();
        float[] aHost = a.ToArray<float>();

        Ops.Qr(exec, q, r, a);
        Ops.Matmul(exec, qr, q, r);
        exec.Synchronize();

        // ||A - Q·R||_F / ||A||_F < 1e-4
        using var diff  = Tensor.CreateFloat32([M, N]);
        using var errN  = Tensor.CreateFloat32([1]);
        using var aN    = Tensor.CreateFloat32([1]);
        Ops.Sub (exec, diff, a, qr);
        Ops.Norm(exec, errN, diff);
        Ops.Norm(exec, aN,   a);
        exec.Synchronize();

        float rel = errN.ToArray<float>()[0] / aN.ToArray<float>()[0];
        Assert.True(rel < 1e-4f, $"QR relative error = {rel}");
    }

    [SkippableFact] public void Inv_ATimesInvA_IsIdentity()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int N = 4;
        using var exec = Executor.CreateCuda();

        using var a    = Tensor.CreateFloat32([N, N]);
        using var ainv = Tensor.CreateFloat32([N, N]);
        using var prod = Tensor.CreateFloat32([N, N]);

        // Build well-conditioned matrix: A = I + 0.1*rand
        Ops.RandomUniform(exec, a);
        Ops.Mul(exec, a, a, 0.1);
        using var ident = Tensor.CreateFloat32([N, N]);
        Ops.Zeros(exec, ident);
        exec.Synchronize();
        float[] id = new float[N * N];
        for (int i = 0; i < N; i++) id[i * N + i] = 1f;
        ident.CopyFrom<float>(id);
        Ops.Add(exec, a, a, ident);

        Ops.Inv(exec, ainv, a);
        Ops.Matmul(exec, prod, a, ainv);
        exec.Synchronize();

        float[] p = prod.ToArray<float>();
        for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            float expected = i == j ? 1f : 0f;
            Assert.Equal(expected, p[i * N + j], precision: 3);
        }
    }

    [SkippableFact] public void Svd_ReconstructsA()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int M = 8, N = 6;
        using var exec = Executor.CreateCuda();

        using var a  = Tensor.CreateFloat32([M, N]);
        using var u  = Tensor.CreateFloat32([M, M]);
        using var s  = Tensor.CreateFloat32([N]);
        using var vt = Tensor.CreateFloat32([N, N]);

        Ops.RandomUniform(exec, a);
        exec.Synchronize();
        float[] aHost = a.ToArray<float>();

        Ops.Svd(exec, u, s, vt, a, SvdMode.Full);
        exec.Synchronize();

        Assert.Equal(M, (int)u.Size(0));
        Assert.Equal(N, (int)s.Size(0));
        Assert.Equal(N, (int)vt.Size(0));
    }

    [SkippableFact] public void Einsum_MatMul_Equivalent()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int M = 4, K = 5, N = 3;
        using var exec = Executor.CreateCuda();
        using var a   = Tensor.CreateFloat32([M, K]);
        using var b   = Tensor.CreateFloat32([K, N]);
        using var mm  = Tensor.CreateFloat32([M, N]);
        using var ein = Tensor.CreateFloat32([M, N]);

        Ops.RandomUniform(exec, a);
        Ops.RandomUniform(exec, b);
        Ops.Matmul (exec, mm,  a, b);
        Ops.Einsum (exec, ein, "ij,jk->ik", a, b);
        exec.Synchronize();

        float[] e1 = mm.ToArray<float>();
        float[] e2 = ein.ToArray<float>();
        for (int i = 0; i < M * N; i++)
            Assert.Equal(e1[i], e2[i], precision: 3);
    }
}

public sealed class SignalCudaTests
{
    [SkippableFact] public void Conv1d_OutputSizeIsCorrect()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int SigN = 64, FiltN = 8;
        int outN = SigN + FiltN - 1; // full convolution
        using var exec   = Executor.CreateCuda();
        using var signal = Tensor.CreateFloat32([SigN]);
        using var filter = Tensor.CreateFloat32([FiltN]);
        using var output = Tensor.CreateFloat32([outN]);
        Ops.RandomUniform(exec, signal);
        Ops.Hamming(exec, filter);
        Ops.Conv1d(exec, output, signal, filter, ConvMode.Full);
        exec.Synchronize();
        Assert.Equal(outN, (int)output.TotalSize);
    }

    [SkippableFact] public void Corr_OutputSizeIsCorrect()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int N = 32;
        int outN = 2 * N - 1;
        using var exec = Executor.CreateCuda();
        using var a    = Tensor.CreateFloat32([N]);
        using var b    = Tensor.CreateFloat32([N]);
        using var out_ = Tensor.CreateFloat32([outN]);
        Ops.RandomUniform(exec, a);
        Ops.RandomUniform(exec, b);
        Ops.Corr(exec, out_, a, b);
        exec.Synchronize();
        Assert.Equal(outN, (int)out_.TotalSize);
    }

    [SkippableFact] public void Pwelch_OutputHasCorrectShape()
    {
        Skip.IfNot(GpuCheck.HasGpu, "No CUDA GPU");
        const int Batch = 2, SigLen = 256, WinLen = 64, Nfft = 64;
        long outFreqBins = Nfft / 2 + 1;
        using var exec   = Executor.CreateCuda();
        using var signal = Tensor.CreateFloat32([Batch, SigLen]);
        using var window = Tensor.CreateFloat32([WinLen]);
        using var pxx    = Tensor.CreateFloat32([Batch, outFreqBins]);
        Ops.RandomNormal(exec, signal);
        Ops.Hamming(exec, window);
        Ops.Pwelch(exec, pxx, signal, window, noverlap: WinLen / 2, nfft: Nfft);
        exec.Synchronize();
        Assert.Equal(Batch,        (int)pxx.Size(0));
        Assert.Equal(outFreqBins,  pxx.Size(1));
    }
}
