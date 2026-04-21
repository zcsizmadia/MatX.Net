// Ops.cs – Static entry points mirroring the MatX operator API.
// All operations follow the pattern:
//   Ops.Foo(executor, destination, source...)
// Operations are asynchronous (queued on the CUDA stream); call
// executor.Synchronize() to wait for completion.
using MatX.Net.Native;

namespace MatX.Net;

/// <summary>
/// Element-wise math, reductions, generators, FFT, linear algebra and signal
/// processing – all dispatched through a native <see cref="Executor"/>.
/// </summary>
public static class Ops
{
    // ── Internal shorthand ────────────────────────────────────────────────────
    private static IntPtr H(Tensor t)   => t.Handle;
    private static IntPtr H(Executor e) => e.Handle;

    // ─────────────────────────────────────────────────────────────────────────
    //  Generators
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>Fill <paramref name="dst"/> with zeros.</summary>
    public static void Zeros         (Executor exec, Tensor dst) => NativeLib.Check(NativeLib.FillZeros        (H(exec), H(dst)));
    /// <summary>Fill <paramref name="dst"/> with ones.</summary>
    public static void Ones          (Executor exec, Tensor dst) => NativeLib.Check(NativeLib.FillOnes         (H(exec), H(dst)));
    /// <summary>Fill <paramref name="dst"/> with U[0,1) random values.</summary>
    public static void RandomUniform (Executor exec, Tensor dst) => NativeLib.Check(NativeLib.FillRandomUniform(H(exec), H(dst)));
    /// <summary>Fill <paramref name="dst"/> with N(0,1) random values.</summary>
    public static void RandomNormal  (Executor exec, Tensor dst) => NativeLib.Check(NativeLib.FillRandomNormal (H(exec), H(dst)));
    /// <summary>Fill a 1-D tensor with evenly spaced values from <paramref name="start"/> to <paramref name="stop"/>.</summary>
    public static void Linspace      (Executor exec, Tensor dst, double start, double stop) => NativeLib.Check(NativeLib.FillLinspace(H(exec), H(dst), start, stop));
    /// <summary>Fill a 1-D tensor with an arithmetic sequence.</summary>
    public static void Range         (Executor exec, Tensor dst, double start, double step) => NativeLib.Check(NativeLib.FillRange   (H(exec), H(dst), start, step));
    /// <summary>Fill a 1-D tensor with FFT sample frequencies.</summary>
    public static void FftFreq       (Executor exec, Tensor dst, double sampleRate = 1.0)  => NativeLib.Check(NativeLib.FillFftFreq  (H(exec), H(dst), sampleRate));
    /// <summary>Fill a 1-D tensor with a Hamming window.</summary>
    public static void Hamming       (Executor exec, Tensor dst) => NativeLib.Check(NativeLib.FillHamming      (H(exec), H(dst)));
    /// <summary>Fill a 1-D tensor with a Hanning window.</summary>
    public static void Hanning       (Executor exec, Tensor dst) => NativeLib.Check(NativeLib.FillHanning      (H(exec), H(dst)));
    /// <summary>Fill a 1-D tensor with a Blackman window.</summary>
    public static void Blackman      (Executor exec, Tensor dst) => NativeLib.Check(NativeLib.FillBlackman     (H(exec), H(dst)));
    /// <summary>Fill a 1-D tensor with a Bartlett window.</summary>
    public static void Bartlett      (Executor exec, Tensor dst) => NativeLib.Check(NativeLib.FillBartlett     (H(exec), H(dst)));

    // ─────────────────────────────────────────────────────────────────────────
    //  Unary element-wise
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary> Computes the absolute value of each element in a tensor.</summary>
    public static void Abs    (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Abs    (H(e), H(dst), H(a)));
    /// <summary>Computes the element-wise square of the absolute value of a tensor.</summary>
    public static void Abs2   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Abs2   (H(e), H(dst), H(a)));
    /// <summary>Computes the square root of each element in a tensor.</summary>
    public static void Sqrt   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Sqrt   (H(e), H(dst), H(a)));
    /// <summary>Computes the reciprocal of each element in a tensor.</summary>
    public static void Exp    (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Exp    (H(e), H(dst), H(a)));
    /// <summary>Computes the natural logarithm of each element in a tensor.</summary>
    public static void Log    (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Log    (H(e), H(dst), H(a)));
    /// <summary>Computes the base-2 logarithm of each element in a tensor.</summary>
    public static void Log2   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Log2   (H(e), H(dst), H(a)));
    /// <summary>Computes the base-10 logarithm of each element in a tensor.</summary>
    public static void Log10  (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Log10  (H(e), H(dst), H(a)));
    /// <summary>Computes the sine of each element in a tensor.</summary>
    public static void Sin    (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Sin    (H(e), H(dst), H(a)));
    /// <summary>Computes the cosine of each element in a tensor.</summary>
    public static void Cos    (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Cos    (H(e), H(dst), H(a)));
    /// <summary>Computes the tangent of each element in a tensor.</summary>
    public static void Tan    (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Tan    (H(e), H(dst), H(a)));
    /// <summary>Computes the arcsine of each element in a tensor.</summary>
    public static void Asin   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Asin   (H(e), H(dst), H(a)));
    /// <summary>Computes the arccosine of each element in a tensor.</summary>
    public static void Acos   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Acos   (H(e), H(dst), H(a)));
    /// <summary>Computes the arctangent of each element in a tensor.</summary>
    public static void Atan   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Atan   (H(e), H(dst), H(a)));
    /// <summary>Computes the hyperbolic sine of each element in a tensor.</summary>
    public static void Sinh   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Sinh   (H(e), H(dst), H(a)));
    /// <summary>Computes the hyperbolic cosine of each element in a tensor.</summary>
    public static void Cosh   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Cosh   (H(e), H(dst), H(a)));
    /// <summary>Computes the hyperbolic tangent of each element in a tensor.</summary>
    public static void Tanh   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Tanh   (H(e), H(dst), H(a)));
    /// <summary>Computes the ceiling of each element in a tensor.</summary>
    public static void Ceil   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Ceil   (H(e), H(dst), H(a)));
    /// <summary>Computes the floor of each element in a tensor.</summary>
    public static void Floor  (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Floor  (H(e), H(dst), H(a)));
    /// <summary>Computes the rounding of each element in a tensor.</summary>
    public static void Round  (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Round  (H(e), H(dst), H(a)));
    /// <summary>Computes the sign of each element in a tensor.</summary>
    public static void Sign   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Sign   (H(e), H(dst), H(a)));
    /// <summary>Computes the negation of each element in a tensor.</summary>
    public static void Neg    (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Neg    (H(e), H(dst), H(a)));
    public static void Conj   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Conj   (H(e), H(dst), H(a)));
    public static void Real   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Real   (H(e), H(dst), H(a)));
    public static void Imag   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Imag   (H(e), H(dst), H(a)));
    public static void NormCdf(Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.NormCdf(H(e), H(dst), H(a)));
    public static void Erf    (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Erf    (H(e), H(dst), H(a)));
    public static void Erfc   (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Erfc   (H(e), H(dst), H(a)));

    // ─────────────────────────────────────────────────────────────────────────
    //  Binary tensor–tensor
    // ─────────────────────────────────────────────────────────────────────────

    public static void Add    (Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Add    (H(e), H(dst), H(a), H(b)));
    public static void Sub    (Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Sub    (H(e), H(dst), H(a), H(b)));
    public static void Mul    (Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Mul    (H(e), H(dst), H(a), H(b)));
    public static void Div    (Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Div    (H(e), H(dst), H(a), H(b)));
    public static void Pow    (Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Pow    (H(e), H(dst), H(a), H(b)));
    public static void Maximum(Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Maximum(H(e), H(dst), H(a), H(b)));
    public static void Minimum(Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Minimum(H(e), H(dst), H(a), H(b)));

    // ── Scalar variants ───────────────────────────────────────────────────────
    public static void Add(Executor e, Tensor dst, Tensor a, double scalar) => NativeLib.Check(NativeLib.AddScalar(H(e), H(dst), H(a), scalar));
    public static void Sub(Executor e, Tensor dst, Tensor a, double scalar) => NativeLib.Check(NativeLib.SubScalar(H(e), H(dst), H(a), scalar));
    public static void Mul(Executor e, Tensor dst, Tensor a, double scalar) => NativeLib.Check(NativeLib.MulScalar(H(e), H(dst), H(a), scalar));
    public static void Div(Executor e, Tensor dst, Tensor a, double scalar) => NativeLib.Check(NativeLib.DivScalar(H(e), H(dst), H(a), scalar));
    public static void Pow(Executor e, Tensor dst, Tensor a, double scalar) => NativeLib.Check(NativeLib.PowScalar(H(e), H(dst), H(a), scalar));

    // ─────────────────────────────────────────────────────────────────────────
    //  Reductions   (axis == -1 → reduce all dimensions)
    // ─────────────────────────────────────────────────────────────────────────

    public static void Sum   (Executor e, Tensor dst, Tensor src, int axis = -1) => NativeLib.Check(NativeLib.Sum  (H(e), H(dst), H(src), axis));
    public static void Mean  (Executor e, Tensor dst, Tensor src, int axis = -1) => NativeLib.Check(NativeLib.Mean (H(e), H(dst), H(src), axis));
    public static void Max   (Executor e, Tensor dst, Tensor src, int axis = -1) => NativeLib.Check(NativeLib.MaxR (H(e), H(dst), H(src), axis));
    public static void Min   (Executor e, Tensor dst, Tensor src, int axis = -1) => NativeLib.Check(NativeLib.MinR (H(e), H(dst), H(src), axis));
    public static void Prod  (Executor e, Tensor dst, Tensor src, int axis = -1) => NativeLib.Check(NativeLib.Prod (H(e), H(dst), H(src), axis));
    public static void Var   (Executor e, Tensor dst, Tensor src, int axis = -1) => NativeLib.Check(NativeLib.Var  (H(e), H(dst), H(src), axis));
    public static void Std   (Executor e, Tensor dst, Tensor src, int axis = -1) => NativeLib.Check(NativeLib.Stdd (H(e), H(dst), H(src), axis));
    public static void Any   (Executor e, Tensor dst, Tensor src)                => NativeLib.Check(NativeLib.Any  (H(e), H(dst), H(src)));
    public static void Norm  (Executor e, Tensor dst, Tensor src)                => NativeLib.Check(NativeLib.Norm (H(e), H(dst), H(src)));
    public static void Trace (Executor e, Tensor dst, Tensor src)                => NativeLib.Check(NativeLib.Trace(H(e), H(dst), H(src)));
    public static void CumSum(Executor e, Tensor dst, Tensor src, int axis = 0)  => NativeLib.Check(NativeLib.CumSum(H(e), H(dst), H(src), axis));
    public static void Sort  (Executor e, Tensor dst, Tensor src, SortDir dir = SortDir.Ascending) => NativeLib.Check(NativeLib.Sort(H(e), H(dst), H(src), dir));

    // ─────────────────────────────────────────────────────────────────────────
    //  FFT
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>1-D complex-to-complex FFT. <paramref name="nfft"/>==0 uses source length.</summary>
    public static void Fft     (Executor e, Tensor dst, Tensor src, long nfft = 0, FftNorm norm = FftNorm.Backward) => NativeLib.Check(NativeLib.Fft     (H(e), H(dst), H(src), nfft, norm));
    public static void Ifft    (Executor e, Tensor dst, Tensor src, long nfft = 0, FftNorm norm = FftNorm.Backward) => NativeLib.Check(NativeLib.Ifft    (H(e), H(dst), H(src), nfft, norm));
    public static void Fft2    (Executor e, Tensor dst, Tensor src, FftNorm norm = FftNorm.Backward)                => NativeLib.Check(NativeLib.Fft2    (H(e), H(dst), H(src), norm));
    public static void Ifft2   (Executor e, Tensor dst, Tensor src, FftNorm norm = FftNorm.Backward)                => NativeLib.Check(NativeLib.Ifft2   (H(e), H(dst), H(src), norm));
    public static void Rfft    (Executor e, Tensor dst, Tensor src, long nfft = 0, FftNorm norm = FftNorm.Backward) => NativeLib.Check(NativeLib.Rfft    (H(e), H(dst), H(src), nfft, norm));
    public static void Irfft   (Executor e, Tensor dst, Tensor src, long nfft = 0, FftNorm norm = FftNorm.Backward) => NativeLib.Check(NativeLib.Irfft   (H(e), H(dst), H(src), nfft, norm));
    public static void FftShift (Executor e, Tensor dst, Tensor src)                                                => NativeLib.Check(NativeLib.FftShift (H(e), H(dst), H(src)));
    public static void IFftShift(Executor e, Tensor dst, Tensor src)                                                => NativeLib.Check(NativeLib.IFftShift(H(e), H(dst), H(src)));
    public static void Dct     (Executor e, Tensor dst, Tensor src)                                                 => NativeLib.Check(NativeLib.Dct     (H(e), H(dst), H(src)));

    // ─────────────────────────────────────────────────────────────────────────
    //  Linear algebra
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>General matrix multiply (GEMM). Supports rank-2 and batched rank-3.</summary>
    public static void Matmul   (Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Matmul   (H(e), H(dst), H(a), H(b)));
    public static void Matvec   (Executor e, Tensor dst, Tensor a, Tensor x) => NativeLib.Check(NativeLib.Matvec   (H(e), H(dst), H(a), H(x)));
    public static void Outer    (Executor e, Tensor dst, Tensor a, Tensor b) => NativeLib.Check(NativeLib.Outer    (H(e), H(dst), H(a), H(b)));
    public static void Transpose(Executor e, Tensor dst, Tensor src)         => NativeLib.Check(NativeLib.Transpose(H(e), H(dst), H(src)));
    public static void Hermitian(Executor e, Tensor dst, Tensor src)         => NativeLib.Check(NativeLib.Hermitian(H(e), H(dst), H(src)));

    /// <summary>SVD: dst = U·diag(S)·Vt. All output tensors must be pre-allocated.</summary>
    public static void Svd  (Executor e, Tensor u, Tensor s, Tensor vt, Tensor a, SvdMode mode = SvdMode.Reduced) => NativeLib.Check(NativeLib.Svd  (H(e), H(u), H(s), H(vt), H(a), mode));
    /// <summary>Truncated SVD via power iteration.</summary>
    public static void SvdPi(Executor e, Tensor u, Tensor s, Tensor vt, Tensor a, int iterations = 10, int k = -1) => NativeLib.Check(NativeLib.SvdPi(H(e), H(u), H(s), H(vt), H(a), iterations, k == -1 ? (int)a.Size(1) : k));
    public static void Qr   (Executor e, Tensor q, Tensor r,   Tensor a) => NativeLib.Check(NativeLib.Qr   (H(e), H(q), H(r),  H(a)));
    public static void Lu   (Executor e, Tensor l, Tensor u, Tensor piv, Tensor a) => NativeLib.Check(NativeLib.Lu   (H(e), H(l), H(u), H(piv), H(a)));
    public static void Chol (Executor e, Tensor l,   Tensor a) => NativeLib.Check(NativeLib.Chol (H(e), H(l),  H(a)));
    public static void Eig  (Executor e, Tensor vec, Tensor val, Tensor a) => NativeLib.Check(NativeLib.Eig  (H(e), H(vec), H(val), H(a)));
    public static void Solve (Executor e, Tensor x, Tensor a, Tensor b)    => NativeLib.Check(NativeLib.Solve(H(e), H(x), H(a), H(b)));
    public static void Inv  (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Inv  (H(e), H(dst), H(a)));
    public static void Pinv (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Pinv (H(e), H(dst), H(a)));
    public static void Det  (Executor e, Tensor dst, Tensor a) => NativeLib.Check(NativeLib.Det  (H(e), H(dst), H(a)));

    /// <summary>Einstein summation for two operands, e.g. <c>"ij,jk->ik"</c>.</summary>
    public static void Einsum(Executor e, Tensor dst, string subscripts, Tensor a, Tensor b)
        => NativeLib.Check(NativeLib.Einsum2(H(e), H(dst), subscripts, H(a), H(b)));

    // ─────────────────────────────────────────────────────────────────────────
    //  Signal processing
    // ─────────────────────────────────────────────────────────────────────────

    public static void Conv1d(Executor e, Tensor dst, Tensor signal, Tensor filter, ConvMode mode = ConvMode.Full)
        => NativeLib.Check(NativeLib.Conv1d(H(e), H(dst), H(signal), H(filter), mode));

    public static void Conv2d(Executor e, Tensor dst, Tensor signal, Tensor filter, ConvMode mode = ConvMode.Full)
        => NativeLib.Check(NativeLib.Conv2d(H(e), H(dst), H(signal), H(filter), mode));

    public static void Corr(Executor e, Tensor dst, Tensor a, Tensor b)
        => NativeLib.Check(NativeLib.Corr(H(e), H(dst), H(a), H(b)));

    /// <summary>
    /// Welch power spectral density estimate.
    /// <paramref name="pxx"/> shape: (batch, nfft/2+1).
    /// <paramref name="signal"/> shape: (batch, N).
    /// <paramref name="window"/> shape: (windowSize,).
    /// </summary>
    public static void Pwelch(Executor e, Tensor pxx, Tensor signal,
        Tensor window, long noverlap, long nfft)
        => NativeLib.Check(NativeLib.Pwelch(H(e), H(pxx), H(signal), H(window), noverlap, nfft));

    /// <summary>Polyphase rational-rate resampler.</summary>
    public static void ResamplePoly(Executor e, Tensor dst, Tensor src,
        int up, int down, Tensor? filter = null)
        => NativeLib.Check(NativeLib.ResamplePoly(H(e), H(dst), H(src), up, down,
            filter is null ? IntPtr.Zero : H(filter)));
}
