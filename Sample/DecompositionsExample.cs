// DecompositionsExample.cs
// Demonstrates QR, LU, Cholesky and eigendecomposition on a small matrix.
// Mirrors examples/qr.cu and eigenExample.cu.
namespace MatX.Net.Examples;

public static class DecompositionsExample
{
    public static void Run()
    {
        const int N = 6;

        using var exec = Executor.CreateCuda();

        // ── Build a symmetric positive-definite matrix A = Bᵀ·B + ε·I ───────
        using var B    = Tensor.CreateFloat32([N, N]);
        using var A    = Tensor.CreateFloat32([N, N]);
        using var Bt   = Tensor.CreateFloat32([N, N]);
        using var eps  = Tensor.CreateFloat32([N, N]);
        Ops.RandomUniform(exec, B);
        Ops.Transpose(exec, Bt, B);
        Ops.Matmul   (exec, A,  Bt, B);   // A = Bᵀ·B

        // Add ε·I for numerical stability
        Ops.Zeros(exec, eps);
        exec.Synchronize();
        float[] epsData = new float[N * N];
        for (int i = 0; i < N; i++) epsData[i * N + i] = (float)N;
        eps.CopyFrom<float>(epsData);
        Ops.Add(exec, A, A, eps);

        exec.Synchronize();
        float[] aHost = A.ToArray<float>();

        // ─────────────────────────────────────────────────────────────────────
        // 1. QR
        // ─────────────────────────────────────────────────────────────────────
        using var Q  = Tensor.CreateFloat32([N, N]);
        using var R  = Tensor.CreateFloat32([N, N]);
        using var QR = Tensor.CreateFloat32([N, N]);
        Ops.Qr    (exec, Q, R, A);
        Ops.Matmul(exec, QR, Q, R);
        exec.Synchronize();

        float qrErr = RelativeError(aHost, QR.ToArray<float>(), N * N);
        Console.WriteLine($"  QR  relative error : {qrErr:E3}");

        // ─────────────────────────────────────────────────────────────────────
        // 2. Cholesky  (A must be SPD)
        // ─────────────────────────────────────────────────────────────────────
        using var L   = Tensor.CreateFloat32([N, N]);
        using var Lt  = Tensor.CreateFloat32([N, N]);
        using var LLt = Tensor.CreateFloat32([N, N]);
        Ops.Chol     (exec, L,   A);
        Ops.Transpose(exec, Lt,  L);
        Ops.Matmul   (exec, LLt, L, Lt);
        exec.Synchronize();

        float cholErr = RelativeError(aHost, LLt.ToArray<float>(), N * N);
        Console.WriteLine($"  Chol relative error: {cholErr:E3}");

        // ─────────────────────────────────────────────────────────────────────
        // 3. Eigendecomposition  (symmetric A)
        // ─────────────────────────────────────────────────────────────────────
        using var vec = Tensor.CreateFloat32([N, N]);
        using var val = Tensor.CreateFloat32([N]);
        Ops.Eig(exec, vec, val, A);
        exec.Synchronize();

        float[] eigenvalues = val.ToArray<float>();
        bool allPositive = eigenvalues.All(v => v > 0f);
        Console.WriteLine($"  Eigenvalues (all >0 for SPD): {allPositive}");
        Console.WriteLine($"  λ = [{string.Join(", ", eigenvalues.Select(v => v.ToString("F3")))}]");
        Console.WriteLine("  [OK]");
    }

    private static float RelativeError(float[] a, float[] b, int n)
    {
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) { num += (a[i] - b[i]) * (a[i] - b[i]); den += a[i] * a[i]; }
        return (float)Math.Sqrt(num / (den + 1e-30));
    }
}
