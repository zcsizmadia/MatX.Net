// OpsHostTests.cs – Numerically-verified tests using the host executor.
// Every test asserts an exact or analytically-known result, not just structure.
// No CUDA GPU required.
using MatX;
using Xunit;
using static System.MathF;

namespace MatX.Net.Tests;

// =============================================================================
//  Shared helpers
// =============================================================================
file static class H
{
    public static Executor E()                => Executor.CreateHost();
    public static Tensor   F(params long[] s) => Tensor.CreateFloat32(s, MemorySpace.Host);

    public static Tensor Filled(float v, int n)
    {
        var t = F(n);
        t.CopyFrom<float>(Enumerable.Repeat(v, n).ToArray());
        return t;
    }
    public static Tensor From(float[] data)
    {
        var t = F(data.Length);
        t.CopyFrom<float>(data);
        return t;
    }
    // Run a global reduction and return the scalar result.
    public static float Reduce1(float[] data, System.Action<Executor, Tensor, Tensor> op)
    {
        using var e   = E();
        using var src = From(data);
        using var dst = F(1);
        op(e, dst, src);
        e.Synchronize();
        return dst.ToArray<float>()[0];
    }
}

// =============================================================================
//  Generators
// =============================================================================
public sealed class GeneratorCalculationTests
{
    [Fact] public void Zeros_AllExactlyZero()
    {
        using var e = H.E(); using var t = H.F(8);
        Ops.Zeros(e, t); e.Synchronize();
        Assert.Equal(new float[8], t.ToArray<float>());
    }

    [Fact] public void Ones_AllExactlyOne()
    {
        using var e = H.E(); using var t = H.F(8);
        Ops.Ones(e, t); e.Synchronize();
        Assert.All(t.ToArray<float>(), v => Assert.Equal(1f, v));
    }

    [Fact] public void Linspace_FivePoints_CorrectValues()
    {
        // linspace(0, 1, 5) -> [0, 0.25, 0.5, 0.75, 1.0]
        using var e = H.E(); using var t = H.F(5);
        Ops.Linspace(e, t, 0.0, 1.0); e.Synchronize();
        float[] v = t.ToArray<float>();
        Assert.Equal(0.00f, v[0], precision: 5);
        Assert.Equal(0.25f, v[1], precision: 5);
        Assert.Equal(0.50f, v[2], precision: 5);
        Assert.Equal(0.75f, v[3], precision: 5);
        Assert.Equal(1.00f, v[4], precision: 5);
    }

    [Fact] public void Range_StepTwo_CorrectValues()
    {
        // range(start=0, step=2, n=4) -> [0, 2, 4, 6]
        using var e = H.E(); using var t = H.F(4);
        Ops.Range(e, t, 0.0, 2.0); e.Synchronize();
        float[] v = t.ToArray<float>();
        Assert.Equal(0f, v[0], precision: 5);
        Assert.Equal(2f, v[1], precision: 5);
        Assert.Equal(4f, v[2], precision: 5);
        Assert.Equal(6f, v[3], precision: 5);
    }

    [Fact] public void FftFreq_EightPoint_KnownBins()
    {
        // fftfreq(N=8, sr=1): [0, 1/8, 2/8, 3/8, ...]
        using var e = H.E(); using var t = H.F(8);
        Ops.FftFreq(e, t, 1.0); e.Synchronize();
        float[] v = t.ToArray<float>();
        Assert.Equal(0f,     v[0], precision: 5);
        Assert.Equal(1f/8f,  v[1], precision: 5);
        Assert.Equal(2f/8f,  v[2], precision: 5);
        Assert.Equal(3f/8f,  v[3], precision: 5);
    }

    [Fact] public void Hamming_Symmetric_PeakAtCentre()
    {
        using var e = H.E(); using var t = H.F(64);
        Ops.Hamming(e, t); e.Synchronize();
        float[] v = t.ToArray<float>();
        for (int i = 0; i < 32; i++)
            Assert.Equal(v[i], v[63 - i], precision: 4);
        Assert.True(v[0]  < 0.10f, "endpoint should be near 0.08");
        Assert.True(v[31] > 0.90f, "centre should be near 1.0");
    }

    [Fact] public void Hanning_Symmetric_EndsNearZero()
    {
        using var e = H.E(); using var t = H.F(64);
        Ops.Hanning(e, t); e.Synchronize();
        float[] v = t.ToArray<float>();
        for (int i = 0; i < 32; i++)
            Assert.Equal(v[i], v[63 - i], precision: 4);
        Assert.True(Abs(v[0]) < 0.005f, "Hanning endpoint should be near 0");
    }

    [Fact] public void Blackman_Symmetric_EndsNearZero()
    {
        using var e = H.E(); using var t = H.F(64);
        Ops.Blackman(e, t); e.Synchronize();
        float[] v = t.ToArray<float>();
        Assert.True(v[0] < 0.01f);
        for (int i = 0; i < 32; i++)
            Assert.Equal(v[i], v[63 - i], precision: 4);
    }

    [Fact] public void Bartlett_Symmetric_PeakAtCentre()
    {
        using var e = H.E(); using var t = H.F(64);
        Ops.Bartlett(e, t); e.Synchronize();
        float[] v = t.ToArray<float>();
        for (int i = 0; i < 32; i++)
            Assert.Equal(v[i], v[63 - i], precision: 4);
        Assert.True(v[32] > v[0], "Bartlett peak should be in the middle");
    }

    [Fact] public void RandomUniform_ValuesInUnitInterval()
    {
        using var e = H.E(); using var t = H.F(512);
        Ops.RandomUniform(e, t); e.Synchronize();
        Assert.All(t.ToArray<float>(), x => Assert.InRange(x, 0f, 1f));
    }
}

// =============================================================================
//  Unary ops – exact analytic values
// =============================================================================
public sealed class UnaryCalculationTests
{
    // Apply a unary op to a single scalar and return the result.
    private static float Apply(float input, System.Action<Executor, Tensor, Tensor> op)
    {
        using var e   = H.E();
        using var src = H.Filled(input, 1);
        using var dst = H.F(1);
        op(e, dst, src);
        e.Synchronize();
        return dst.ToArray<float>()[0];
    }

    [Fact] public void Abs_Negative_BecomesPositive()   => Assert.Equal(7f,    Apply(-7f,    Ops.Abs),     precision: 5);
    [Fact] public void Abs_Positive_Unchanged()         => Assert.Equal(7f,    Apply( 7f,    Ops.Abs),     precision: 5);
    [Fact] public void Abs2_Negative_SquaredMagnitude() => Assert.Equal(49f,   Apply(-7f,    Ops.Abs2),    precision: 4);
    [Fact] public void Neg_Positive_Flipped()           => Assert.Equal(-5f,   Apply( 5f,    Ops.Neg),     precision: 5);
    [Fact] public void Neg_Negative_Flipped()           => Assert.Equal( 5f,   Apply(-5f,    Ops.Neg),     precision: 5);
    [Fact] public void Sqrt_16_Is4()                    => Assert.Equal(4f,    Apply(16f,    Ops.Sqrt),    precision: 5);
    [Fact] public void Sqrt_0_Is0()                     => Assert.Equal(0f,    Apply( 0f,    Ops.Sqrt),    precision: 5);
    [Fact] public void Exp_0_Is1()                      => Assert.Equal(1f,    Apply( 0f,    Ops.Exp),     precision: 5);
    [Fact] public void Exp_1_IsE()                      => Assert.Equal(E,     Apply( 1f,    Ops.Exp),     precision: 4);
    [Fact] public void Log_1_Is0()                      => Assert.Equal(0f,    Apply( 1f,    Ops.Log),     precision: 5);
    [Fact] public void Log_E_Is1()                      => Assert.Equal(1f,    Apply( E,     Ops.Log),     precision: 5);
    [Fact] public void Log2_8_Is3()                     => Assert.Equal(3f,    Apply( 8f,    Ops.Log2),    precision: 5);
    [Fact] public void Log2_1_Is0()                     => Assert.Equal(0f,    Apply( 1f,    Ops.Log2),    precision: 5);
    [Fact] public void Log10_100_Is2()                  => Assert.Equal(2f,    Apply(100f,   Ops.Log10),   precision: 5);
    [Fact] public void Log10_1_Is0()                    => Assert.Equal(0f,    Apply(  1f,   Ops.Log10),   precision: 5);
    [Fact] public void Sin_HalfPi_Is1()                 => Assert.Equal(1f,    Apply(PI/2f,  Ops.Sin),     precision: 5);
    [Fact] public void Sin_Pi_IsNearZero()              => Assert.Equal(0f,    Apply(PI,     Ops.Sin),     precision: 5);
    [Fact] public void Sin_Zero_IsZero()                => Assert.Equal(0f,    Apply( 0f,    Ops.Sin),     precision: 5);
    [Fact] public void Cos_Zero_IsOne()                 => Assert.Equal(1f,    Apply( 0f,    Ops.Cos),     precision: 5);
    [Fact] public void Cos_Pi_IsMinus1()                => Assert.Equal(-1f,   Apply( PI,    Ops.Cos),     precision: 5);
    [Fact] public void Cos_HalfPi_IsNearZero()          => Assert.Equal(0f,    Apply(PI/2f,  Ops.Cos),     precision: 5);
    [Fact] public void Tan_PiOver4_Is1()                => Assert.Equal(1f,    Apply(PI/4f,  Ops.Tan),     precision: 5);
    [Fact] public void Asin_1_IsHalfPi()                => Assert.Equal(PI/2f, Apply( 1f,    Ops.Asin),    precision: 5);
    [Fact] public void Acos_1_IsZero()                  => Assert.Equal(0f,    Apply( 1f,    Ops.Acos),    precision: 5);
    [Fact] public void Atan_1_IsPiOver4()               => Assert.Equal(PI/4f, Apply( 1f,    Ops.Atan),    precision: 5);
    [Fact] public void Sinh_Zero_IsZero()               => Assert.Equal(0f,    Apply( 0f,    Ops.Sinh),    precision: 5);
    [Fact] public void Cosh_Zero_IsOne()                => Assert.Equal(1f,    Apply( 0f,    Ops.Cosh),    precision: 5);
    [Fact] public void Tanh_Zero_IsZero()               => Assert.Equal(0f,    Apply( 0f,    Ops.Tanh),    precision: 5);
    [Fact] public void Ceil_1p2_Is2()                   => Assert.Equal(2f,    Apply(1.2f,   Ops.Ceil),    precision: 5);
    [Fact] public void Ceil_Neg0p8_IsZero()             => Assert.Equal(0f,    Apply(-0.8f,  Ops.Ceil),    precision: 5);
    [Fact] public void Floor_1p9_Is1()                  => Assert.Equal(1f,    Apply(1.9f,   Ops.Floor),   precision: 5);
    [Fact] public void Floor_Neg1p1_IsMinus2()          => Assert.Equal(-2f,   Apply(-1.1f,  Ops.Floor),   precision: 5);
    [Fact] public void Round_2p5_Is3()                  => Assert.Equal(3f,    Apply(2.5f,   Ops.Round),   precision: 5);
    [Fact] public void Round_2p4_Is2()                  => Assert.Equal(2f,    Apply(2.4f,   Ops.Round),   precision: 5);
    [Fact] public void Sign_Positive_Is1()              => Assert.Equal( 1f,   Apply( 3f,    Ops.Sign),    precision: 5);
    [Fact] public void Sign_Negative_IsMinus1()         => Assert.Equal(-1f,   Apply(-3f,    Ops.Sign),    precision: 5);
    [Fact] public void Sign_Zero_IsZero()               => Assert.Equal( 0f,   Apply( 0f,    Ops.Sign),    precision: 5);
    [Fact] public void Erf_Zero_IsZero()                => Assert.Equal(0f,    Apply( 0f,    Ops.Erf),     precision: 5);
    [Fact] public void Erfc_Zero_IsOne()                => Assert.Equal(1f,    Apply( 0f,    Ops.Erfc),    precision: 5);
    [Fact] public void Erf_Large_NearOne()              => Assert.True(        Apply( 5f,    Ops.Erf) > 0.9999f);
    [Fact] public void NormCdf_Zero_IsHalf()            => Assert.Equal(0.5f,  Apply( 0f,    Ops.NormCdf), precision: 5);
    [Fact] public void NormCdf_VeryLarge_NearOne()      => Assert.True(        Apply(10f,    Ops.NormCdf) > 0.999f);
    [Fact] public void NormCdf_VeryNegative_NearZero()  => Assert.True(        Apply(-10f,   Ops.NormCdf) < 0.001f);
}

// =============================================================================
//  Binary ops – exact element-wise results
// =============================================================================
public sealed class BinaryCalculationTests
{
    [Fact] public void Add_KnownVectors()
    {
        // [1,2,3] + [4,5,6] = [5,7,9]
        using var e = H.E();
        using var a = H.From([1f, 2f, 3f]);
        using var b = H.From([4f, 5f, 6f]);
        using var d = H.F(3);
        Ops.Add(e, d, a, b); e.Synchronize();
        Assert.Equal([5f, 7f, 9f], d.ToArray<float>());
    }

    [Fact] public void Sub_KnownVectors()
    {
        // [5,5,5] - [1,2,3] = [4,3,2]
        using var e = H.E();
        using var a = H.From([5f, 5f, 5f]);
        using var b = H.From([1f, 2f, 3f]);
        using var d = H.F(3);
        Ops.Sub(e, d, a, b); e.Synchronize();
        Assert.Equal([4f, 3f, 2f], d.ToArray<float>());
    }

    [Fact] public void Mul_KnownVectors()
    {
        // [2,3,4] * [3,4,5] = [6,12,20]
        using var e = H.E();
        using var a = H.From([2f, 3f, 4f]);
        using var b = H.From([3f, 4f, 5f]);
        using var d = H.F(3);
        Ops.Mul(e, d, a, b); e.Synchronize();
        Assert.Equal([6f, 12f, 20f], d.ToArray<float>());
    }

    [Fact] public void Div_KnownVectors()
    {
        // [6,9,12] / [2,3,4] = [3,3,3]
        using var e = H.E();
        using var a = H.From([6f, 9f, 12f]);
        using var b = H.From([2f, 3f, 4f]);
        using var d = H.F(3);
        Ops.Div(e, d, a, b); e.Synchronize();
        Assert.All(d.ToArray<float>(), v => Assert.Equal(3f, v, precision: 5));
    }

    [Fact] public void Pow_TensorTensor_KnownValues()
    {
        // [2,3,4]^[2,2,2] = [4,9,16]
        using var e = H.E();
        using var a = H.From([2f, 3f, 4f]);
        using var b = H.From([2f, 2f, 2f]);
        using var d = H.F(3);
        Ops.Pow(e, d, a, b); e.Synchronize();
        float[] v = d.ToArray<float>();
        Assert.Equal( 4f, v[0], precision: 4);
        Assert.Equal( 9f, v[1], precision: 4);
        Assert.Equal(16f, v[2], precision: 4);
    }

    [Fact] public void AddScalar_ShiftsAllByTen()
    {
        // [1,2,3] + 10 = [11,12,13]
        using var e = H.E();
        using var a = H.From([1f, 2f, 3f]);
        using var d = H.F(3);
        Ops.Add(e, d, a, 10.0); e.Synchronize();
        Assert.Equal([11f, 12f, 13f], d.ToArray<float>());
    }

    [Fact] public void SubScalar_KnownResult()
    {
        // [10,20,30] - 5 = [5,15,25]
        using var e = H.E();
        using var a = H.From([10f, 20f, 30f]);
        using var d = H.F(3);
        Ops.Sub(e, d, a, 5.0); e.Synchronize();
        Assert.Equal([5f, 15f, 25f], d.ToArray<float>());
    }

    [Fact] public void MulScalar_TripleAll()
    {
        // [1,2,3] * 3 = [3,6,9]
        using var e = H.E();
        using var a = H.From([1f, 2f, 3f]);
        using var d = H.F(3);
        Ops.Mul(e, d, a, 3.0); e.Synchronize();
        Assert.Equal([3f, 6f, 9f], d.ToArray<float>());
    }

    [Fact] public void DivScalar_HalvesAll()
    {
        // [2,4,6] / 2 = [1,2,3]
        using var e = H.E();
        using var a = H.From([2f, 4f, 6f]);
        using var d = H.F(3);
        Ops.Div(e, d, a, 2.0); e.Synchronize();
        Assert.Equal([1f, 2f, 3f], d.ToArray<float>());
    }

    [Fact] public void PowScalar_Squaring()
    {
        // [2,3,4]^2 = [4,9,16]
        using var e = H.E();
        using var a = H.From([2f, 3f, 4f]);
        using var d = H.F(3);
        Ops.Pow(e, d, a, 2.0); e.Synchronize();
        float[] v = d.ToArray<float>();
        Assert.Equal( 4f, v[0], precision: 4);
        Assert.Equal( 9f, v[1], precision: 4);
        Assert.Equal(16f, v[2], precision: 4);
    }

    [Fact] public void PowScalar_Cubing()
    {
        // [2,3]^3 = [8,27]
        using var e = H.E();
        using var a = H.From([2f, 3f]);
        using var d = H.F(2);
        Ops.Pow(e, d, a, 3.0); e.Synchronize();
        float[] v = d.ToArray<float>();
        Assert.Equal( 8f, v[0], precision: 3);
        Assert.Equal(27f, v[1], precision: 3);
    }

    [Fact] public void Maximum_ElementWise_KnownResult()
    {
        // max([1,5,3],[4,2,6]) = [4,5,6]
        using var e = H.E();
        using var a = H.From([1f, 5f, 3f]);
        using var b = H.From([4f, 2f, 6f]);
        using var d = H.F(3);
        Ops.Maximum(e, d, a, b); e.Synchronize();
        Assert.Equal([4f, 5f, 6f], d.ToArray<float>());
    }

    [Fact] public void Minimum_ElementWise_KnownResult()
    {
        // min([1,5,3],[4,2,6]) = [1,2,3]
        using var e = H.E();
        using var a = H.From([1f, 5f, 3f]);
        using var b = H.From([4f, 2f, 6f]);
        using var d = H.F(3);
        Ops.Minimum(e, d, a, b); e.Synchronize();
        Assert.Equal([1f, 2f, 3f], d.ToArray<float>());
    }
}

// =============================================================================
//  Reductions – exact analytic values
// =============================================================================
public sealed class ReductionCalculationTests
{
    [Fact] public void Sum_1to5_Is15()
        => Assert.Equal(15f, H.Reduce1([1f,2f,3f,4f,5f], (e,d,s) => Ops.Sum(e,d,s)), precision: 5);

    [Fact] public void Sum_AllZeros_IsZero()
        => Assert.Equal(0f,  H.Reduce1([0f,0f,0f],       (e,d,s) => Ops.Sum(e,d,s)), precision: 5);

    [Fact] public void Sum_NegativeValues()
        => Assert.Equal(-6f, H.Reduce1([-1f,-2f,-3f],    (e,d,s) => Ops.Sum(e,d,s)), precision: 5);

    [Fact] public void Mean_1to5_Is3()
        => Assert.Equal(3f,  H.Reduce1([1f,2f,3f,4f,5f], (e,d,s) => Ops.Mean(e,d,s)), precision: 5);

    [Fact] public void Mean_AllSame_IsTheSame()
        => Assert.Equal(7f,  H.Reduce1([7f,7f,7f,7f],    (e,d,s) => Ops.Mean(e,d,s)), precision: 5);

    [Fact] public void Max_MixedValues_IsLargest()
        => Assert.Equal(9f,  H.Reduce1([3f,1f,4f,1f,5f,9f,2f,6f], (e,d,s) => Ops.Max(e,d,s)), precision: 5);

    [Fact] public void Max_AllNegative_IsLeastNegative()
        => Assert.Equal(-1f, H.Reduce1([-3f,-1f,-4f,-2f], (e,d,s) => Ops.Max(e,d,s)), precision: 5);

    [Fact] public void Min_MixedValues_IsSmallest()
        => Assert.Equal(1f,  H.Reduce1([3f,1f,4f,1f,5f,9f,2f,6f], (e,d,s) => Ops.Min(e,d,s)), precision: 5);

    [Fact] public void Min_AllNegative_IsMostNegative()
        => Assert.Equal(-4f, H.Reduce1([-3f,-1f,-4f,-2f], (e,d,s) => Ops.Min(e,d,s)), precision: 5);

    [Fact] public void Prod_1to4_Is24()
        => Assert.Equal(24f, H.Reduce1([1f,2f,3f,4f],     (e,d,s) => Ops.Prod(e,d,s)), precision: 4);

    [Fact] public void Prod_PowersOf2_Is64()
        => Assert.Equal(64f, H.Reduce1([2f,4f,8f],         (e,d,s) => Ops.Prod(e,d,s)), precision: 4);

    [Fact] public void Norm_3_4_Vector_Is5()
        => Assert.Equal(5f,  H.Reduce1([3f,4f],            (e,d,s) => Ops.Norm(e,d,s)), precision: 4);

    [Fact] public void Norm_UnitVector_Is1()
        => Assert.Equal(1f,  H.Reduce1([1f,0f,0f],         (e,d,s) => Ops.Norm(e,d,s)), precision: 5);

    [Fact] public void Norm_ZeroVector_Is0()
        => Assert.Equal(0f,  H.Reduce1([0f,0f,0f],         (e,d,s) => Ops.Norm(e,d,s)), precision: 5);

    [Fact] public void Norm_345_Is_Sqrt50()
    {
        float expected = Sqrt(3*3 + 4*4 + 5*5);
        Assert.Equal(expected, H.Reduce1([3f,4f,5f], (e,d,s) => Ops.Norm(e,d,s)), precision: 4);
    }

    [Fact] public void Trace_2x2Matrix_Is5()
    {
        // [[1,2],[3,4]] -> trace = 1+4 = 5
        using var src = H.F(2, 2);
        src.CopyFrom<float>([1f, 2f, 3f, 4f]);
        using var dst = H.F(1); using var e = H.E();
        Ops.Trace(e, dst, src); e.Synchronize();
        Assert.Equal(5f, dst.ToArray<float>()[0], precision: 5);
    }

    [Fact] public void Trace_3x3Identity_Is3()
    {
        using var src = H.F(3, 3);
        src.CopyFrom<float>([1,0,0, 0,1,0, 0,0,1]);
        using var dst = H.F(1); using var e = H.E();
        Ops.Trace(e, dst, src); e.Synchronize();
        Assert.Equal(3f, dst.ToArray<float>()[0], precision: 5);
    }

    [Fact] public void Trace_DiagonalMatrix_Is10()
    {
        // diag([1,2,3,4]) -> trace = 10
        float[] m = new float[16];
        m[0] = 1; m[5] = 2; m[10] = 3; m[15] = 4;
        using var src = H.F(4, 4);
        src.CopyFrom<float>(m);
        using var dst = H.F(1); using var e = H.E();
        Ops.Trace(e, dst, src); e.Synchronize();
        Assert.Equal(10f, dst.ToArray<float>()[0], precision: 5);
    }

    [Fact] public void CumSum_AllOnes_IsNaturalNumbers()
    {
        // cumsum([1,1,1,1]) = [1,2,3,4]
        using var src = H.Filled(1f, 4);
        using var dst = H.F(4); using var e = H.E();
        Ops.CumSum(e, dst, src); e.Synchronize();
        float[] v = dst.ToArray<float>();
        Assert.Equal(1f, v[0], precision: 5);
        Assert.Equal(2f, v[1], precision: 5);
        Assert.Equal(3f, v[2], precision: 5);
        Assert.Equal(4f, v[3], precision: 5);
    }

    [Fact] public void CumSum_ArithmeticSequence()
    {
        // cumsum([1,2,3,4]) = [1,3,6,10]
        using var src = H.From([1f, 2f, 3f, 4f]);
        using var dst = H.F(4); using var e = H.E();
        Ops.CumSum(e, dst, src); e.Synchronize();
        float[] v = dst.ToArray<float>();
        Assert.Equal( 1f, v[0], precision: 5);
        Assert.Equal( 3f, v[1], precision: 5);
        Assert.Equal( 6f, v[2], precision: 5);
        Assert.Equal(10f, v[3], precision: 5);
    }

    [Fact] public void Var_KnownDataset_Is4()
    {
        // pop-var of [2,4,4,4,5,5,7,9]: mean=5, var=4
        float v = H.Reduce1([2f,4f,4f,4f,5f,5f,7f,9f], (e,d,s) => Ops.Var(e,d,s));
        Assert.Equal(4f, v, precision: 3);
    }

    [Fact] public void Std_KnownDataset_Is2()
    {
        float v = H.Reduce1([2f,4f,4f,4f,5f,5f,7f,9f], (e,d,s) => Ops.Std(e,d,s));
        Assert.Equal(2f, v, precision: 3);
    }

    [Fact] public void Sort_Ascending_KnownPermutation()
    {
        using var src = H.From([5f, 1f, 4f, 2f, 3f]);
        using var dst = H.F(5); using var e = H.E();
        Ops.Sort(e, dst, src, SortDir.Ascending); e.Synchronize();
        Assert.Equal([1f, 2f, 3f, 4f, 5f], dst.ToArray<float>());
    }

    [Fact] public void Sort_Descending_KnownPermutation()
    {
        using var src = H.From([5f, 1f, 4f, 2f, 3f]);
        using var dst = H.F(5); using var e = H.E();
        Ops.Sort(e, dst, src, SortDir.Descending); e.Synchronize();
        Assert.Equal([5f, 4f, 3f, 2f, 1f], dst.ToArray<float>());
    }

    [Fact] public void Sort_AlreadySorted_Unchanged()
    {
        using var src = H.From([1f, 2f, 3f, 4f, 5f]);
        using var dst = H.F(5); using var e = H.E();
        Ops.Sort(e, dst, src, SortDir.Ascending); e.Synchronize();
        Assert.Equal([1f, 2f, 3f, 4f, 5f], dst.ToArray<float>());
    }

    [Fact] public void Any_AllZero_IsFalse()
    {
        using var src = H.Filled(0f, 8);
        using var dst = H.F(1); using var e = H.E();
        Ops.Any(e, dst, src); e.Synchronize();
        Assert.Equal(0f, dst.ToArray<float>()[0], precision: 5);
    }

    [Fact] public void Any_OneNonZero_IsTrue()
    {
        using var src = H.From([0f, 0f, 3f, 0f]);
        using var dst = H.F(1); using var e = H.E();
        Ops.Any(e, dst, src); e.Synchronize();
        Assert.NotEqual(0f, dst.ToArray<float>()[0]);
    }
}

// =============================================================================
//  View operations – exact value preservation
// =============================================================================
public sealed class TensorViewCalculationTests
{
    [Fact] public void Flatten_PreservesAllValues()
    {
        float[] data = [1f,2f,3f,4f,5f,6f];
        using var t  = H.F(2, 3);
        t.CopyFrom<float>(data);
        using var fl = t.Flatten();
        Assert.Equal(data, fl.ToArray<float>());
    }

    [Fact] public void Reshape_8to2x4_PreservesValues()
    {
        float[] data = [1f,2f,3f,4f,5f,6f,7f,8f];
        using var t  = H.F(8);
        t.CopyFrom<float>(data);
        using var r  = t.Reshape(2, 4);
        Assert.Equal(data, r.ToArray<float>());
    }

    [Fact] public void Reshape_ChangesShape_SameTotalSize()
    {
        using var e = H.E();
        using var t = H.F(12);
        Ops.Ones(e, t); e.Synchronize();
        using var r = t.Reshape(3, 4);
        Assert.Equal(3L,  r.Size(0));
        Assert.Equal(4L,  r.Size(1));
        Assert.Equal(12L, r.TotalSize);
    }

    [Fact] public void Permute_2D_CorrectNewShape()
    {
        using var t  = H.F(3, 5);
        using var tp = t.Permute(1, 0);
        Assert.Equal(5L, tp.Size(0));
        Assert.Equal(3L, tp.Size(1));
    }

    [Fact] public void Permute_3D_CorrectNewShape()
    {
        using var t  = H.F(2, 3, 4);
        using var tp = t.Permute(2, 0, 1);
        Assert.Equal(4L, tp.Size(0));
        Assert.Equal(2L, tp.Size(1));
        Assert.Equal(3L, tp.Size(2));
    }

    [Fact] public void Slice_ExtractsCorrectSubset()
    {
        // [10,20,30,40,50,60,70,80] sliced [2,6) = [30,40,50,60]
        using var t  = H.F(8);
        t.CopyFrom<float>([10f,20f,30f,40f,50f,60f,70f,80f]);
        using var sl = t.Slice([2], [6]);
        Assert.Equal(4L, sl.Size(0));
        float[] v = sl.ToArray<float>();
        Assert.Equal(30f, v[0], precision: 5);
        Assert.Equal(40f, v[1], precision: 5);
        Assert.Equal(50f, v[2], precision: 5);
        Assert.Equal(60f, v[3], precision: 5);
    }

    [Fact] public void Flatten_After2DShape_CorrectSize()
    {
        using var t  = H.F(4, 6);
        using var fl = t.Flatten();
        Assert.Equal(1,   fl.Rank);
        Assert.Equal(24L, fl.TotalSize);
    }
}
