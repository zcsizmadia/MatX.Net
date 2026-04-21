// BlackScholesExample.cs
// Prices European call options via the Black-Scholes formula on the GPU.
// Mirrors examples/black_scholes.cu.
//
// Formula:
//   d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
//   d2 = d1 − σ√T
//   C  = S·N(d1) − K·e^(−rT)·N(d2)
namespace MatX.Net.Examples;

public static class BlackScholesExample
{
    public static void Run()
    {
        const int N = 1 << 20; // 1 M contracts
        const float r = 0.05f, sigma = 0.2f, T = 1.0f;

        using var exec = Executor.CreateCuda();

        // ── Allocate inputs ──────────────────────────────────────────────────
        using var S = Tensor.CreateFloat32([N]);  // spot prices
        using var K = Tensor.CreateFloat32([N]);  // strike prices

        // Spot ~ U[80,120], strike ~ U[80,120]
        Ops.RandomUniform(exec, S);
        Ops.RandomUniform(exec, K);
        Ops.Mul(exec, S, S, 40.0);  // scale to [0,40]
        Ops.Add(exec, S, S, 80.0);  // shift to [80,120]
        Ops.Mul(exec, K, K, 40.0);
        Ops.Add(exec, K, K, 80.0);

        // ── Intermediate tensors ─────────────────────────────────────────────
        using var d1     = Tensor.CreateFloat32([N]);
        using var d2     = Tensor.CreateFloat32([N]);
        using var nd1    = Tensor.CreateFloat32([N]);
        using var nd2    = Tensor.CreateFloat32([N]);
        using var tmp    = Tensor.CreateFloat32([N]);
        using var call   = Tensor.CreateFloat32([N]);

        double sigmaT    = sigma * Math.Sqrt(T);
        double rT        = r * T;
        double sigSqHalf = sigma * sigma * 0.5;

        // d1 = ln(S/K) / (σ√T)  +  (r + σ²/2)·√T/σ
        Ops.Div(exec, d1, S, K);                // d1 = S/K
        Ops.Log(exec, d1, d1);                  // d1 = ln(S/K)
        Ops.Add(exec, d1, d1, rT + sigSqHalf * T); // d1 += (r + σ²/2)T
        Ops.Div(exec, d1, d1, sigmaT);          // d1 /= σ√T

        // d2 = d1 − σ√T
        Ops.Sub(exec, d2, d1, sigmaT);

        // N(d1), N(d2)
        Ops.NormCdf(exec, nd1, d1);
        Ops.NormCdf(exec, nd2, d2);

        // C = S·N(d1) − K·e^(−rT)·N(d2)
        Ops.Mul(exec, call, S, nd1);            // call  = S·N(d1)
        Ops.Mul(exec, tmp,  K, nd2);            // tmp   = K·N(d2)
        Ops.Mul(exec, tmp,  tmp, Math.Exp(-rT)); // tmp  *= e^(−rT)
        Ops.Sub(exec, call, call, tmp);          // call -= tmp

        exec.Synchronize();

        // ── Summary stats ────────────────────────────────────────────────────
        using var meanCall = Tensor.CreateFloat32([1]);
        using var maxCall  = Tensor.CreateFloat32([1]);
        using var minCall  = Tensor.CreateFloat32([1]);
        Ops.Mean(exec, meanCall, call);
        Ops.Max (exec, maxCall,  call);
        Ops.Min (exec, minCall,  call);
        exec.Synchronize();

        float mean = meanCall.ToArray<float>()[0];
        float max  = maxCall .ToArray<float>()[0];
        float min  = minCall .ToArray<float>()[0];

        Console.WriteLine($"  Contracts priced : {N:N0}");
        Console.WriteLine($"  Call price mean  : {mean:F4}");
        Console.WriteLine($"  Call price range : [{min:F4}, {max:F4}]");
        Console.WriteLine("  [OK]");
    }
}
