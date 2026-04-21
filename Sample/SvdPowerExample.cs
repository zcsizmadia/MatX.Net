// SvdPowerExample.cs
// Truncated SVD via power iteration on a random tall matrix.
// Verifies A ≈ U·diag(S)·Vt by checking ||A - U·diag(S)·Vt||_F / ||A||_F.
// Mirrors examples/svd_power.cu.
namespace MatX.Net.Examples;

public static class SvdPowerExample
{
    public static void Run()
    {
        const int M = 256;   // rows
        const int N = 128;   // cols
        const int K = 16;    // target rank

        using var exec = Executor.CreateCuda();

        // ── Allocate input ──────────────────────────────────────────────────
        using var A = Tensor.CreateFloat32([M, N]);
        A.Name = "A";

        Ops.RandomUniform(exec, A);

        // ── Allocate outputs ────────────────────────────────────────────────
        using var U  = Tensor.CreateFloat32([M, K]);
        using var S  = Tensor.CreateFloat32([K]);
        using var Vt = Tensor.CreateFloat32([K, N]);

        // ── Run SVD power iteration ─────────────────────────────────────────
        Ops.SvdPi(exec, U, S, Vt, A, iterations: 20, k: K);
        exec.Synchronize();

        // ── Reconstruct A_approx = U · diag(S) · Vt ────────────────────────
        // Step 1: scale columns of U by S  →  US = U * S[broadcast]
        using var US  = Tensor.CreateFloat32([M, K]);
        // broadcast S (rank-1) needs manual outer approach:
        // US[:,j] = U[:,j] * S[j]  –  achieved via clone + elementwise mul
        // For simplicity we do it via matmul with diagonal matrix formed from S.
        using var Sdiag = Tensor.CreateFloat32([K, K]);
        Ops.Zeros(exec, Sdiag);
        exec.Synchronize();

        // Fill diagonal of Sdiag with S values (host-side after sync)
        float[] sHost = S.ToArray<float>();
        float[] sdHost = Sdiag.ToArray<float>();
        for (int i = 0; i < K; i++) sdHost[i * K + i] = sHost[i];
        Sdiag.CopyFrom<float>(sdHost);

        // US = U · Sdiag   [M×K] × [K×K] = [M×K]
        Ops.Matmul(exec, US, U, Sdiag);

        // A_approx = US · Vt   [M×K] × [K×N] = [M×N]
        using var A_approx = Tensor.CreateFloat32([M, N]);
        Ops.Matmul(exec, A_approx, US, Vt);

        // ── Compute relative error ||A - A_approx||_F / ||A||_F ─────────────
        using var diff    = Tensor.CreateFloat32([M, N]);
        using var errNorm = Tensor.CreateFloat32([1]);
        using var aNorm   = Tensor.CreateFloat32([1]);

        Ops.Sub (exec, diff,    A, A_approx);
        Ops.Norm(exec, errNorm, diff);
        Ops.Norm(exec, aNorm,   A);
        exec.Synchronize();

        float err = errNorm.ToArray<float>()[0];
        float an  = aNorm.ToArray<float>()[0];
        float rel = err / an;

        Console.WriteLine($"  Matrix shape    : {M}×{N}");
        Console.WriteLine($"  Target rank     : {K}");
        Console.WriteLine($"  ||A||_F         : {an:F4}");
        Console.WriteLine($"  Relative error  : {rel:E4}");
        Console.WriteLine(rel < 0.5f ? "  [OK]" : "  [WARN] relative error higher than expected");
    }
}
