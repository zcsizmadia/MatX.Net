// NativeLib.cs – P/Invoke declarations and native library loader.
// Resolves platform-specific library names automatically.
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace MatX.Net.Native;

/// <summary>
/// Keeps all DllImport declarations in one place.
/// The native library name is resolved per-platform at class-init time.
/// </summary>
internal static partial class NativeLib
{
    // ── Library name ─────────────────────────────────────────────────────────
    private const string LibName = "matxnative";

    static NativeLib()
    {
        // Allow the user to override via environment variable.
        var envPath = Environment.GetEnvironmentVariable("MATX_NATIVE_LIB");
        if (envPath is not null)
            NativeLibrary.Load(envPath);
    }

    // ── Error ─────────────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_get_last_error")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr GetLastError();

    [LibraryImport(LibName, EntryPoint = "matx_clear_error")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void ClearError();

    // ── Tensor life-cycle ─────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_tensor_create")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr TensorCreate(DType dtype, int rank,
        in long shape0, MemorySpace space);

    // shape passed as pointer (unsafe call sites)
    [LibraryImport(LibName, EntryPoint = "matx_tensor_create")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static unsafe partial IntPtr TensorCreatePtr(DType dtype, int rank,
        long* shape, MemorySpace space);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_from_ptr")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static unsafe partial IntPtr TensorFromPtr(DType dtype, int rank,
        long* shape, void* data);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_destroy")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void TensorDestroy(IntPtr handle);

    // ── Tensor metadata ───────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_tensor_rank")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int TensorRank(IntPtr h);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_dtype")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial DType TensorDtype(IntPtr h);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_size")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial long TensorSize(IntPtr h, int dim);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_stride")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial long TensorStride(IntPtr h, int dim);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_total_size")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial long TensorTotalSize(IntPtr h);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_data")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr TensorData(IntPtr h);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_set_name",
        StringMarshalling = StringMarshalling.Utf8)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void TensorSetName(IntPtr h, string name);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_print")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void TensorPrint(IntPtr h);

    // ── Memory helpers ────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_tensor_prefetch_device")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int TensorPrefetchDevice(IntPtr tensor, IntPtr exec);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_copy")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int TensorCopy(IntPtr dst, IntPtr src, IntPtr exec);

    // ── Views ─────────────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_tensor_reshape")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static unsafe partial IntPtr TensorReshape(IntPtr h, int newRank, long* newShape);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_permute")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static unsafe partial IntPtr TensorPermute(IntPtr h, int* axes);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_slice")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static unsafe partial IntPtr TensorSlice(IntPtr h,
        long* starts, long* ends, long* strides);

    [LibraryImport(LibName, EntryPoint = "matx_tensor_flatten")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr TensorFlatten(IntPtr h);

    // ── Executor ──────────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_executor_cuda")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr ExecutorCuda(IntPtr stream);

    [LibraryImport(LibName, EntryPoint = "matx_executor_host")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr ExecutorHost();

    [LibraryImport(LibName, EntryPoint = "matx_executor_destroy")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial void ExecutorDestroy(IntPtr h);

    [LibraryImport(LibName, EntryPoint = "matx_executor_sync")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int ExecutorSync(IntPtr h);

    [LibraryImport(LibName, EntryPoint = "matx_executor_stream")]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial IntPtr ExecutorStream(IntPtr h);

    // ── Unary element-wise ────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_abs")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Abs    (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_abs2")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Abs2   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_sqrt")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Sqrt   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_exp")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Exp    (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_log")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Log    (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_log2")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Log2   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_log10")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Log10  (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_sin")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Sin    (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_cos")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Cos    (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_tan")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Tan    (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_asin")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Asin   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_acos")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Acos   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_atan")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Atan   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_sinh")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Sinh   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_cosh")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Cosh   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_tanh")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Tanh   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_ceil")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Ceil   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_floor")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Floor  (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_round")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Round  (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_sign")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Sign   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_neg")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Neg    (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_conj")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Conj   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_real")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Real   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_imag")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Imag   (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_normcdf")][UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int NormCdf(IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_erf")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Erf    (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_erfc")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Erfc   (IntPtr exec, IntPtr dst, IntPtr a);

    // ── Binary tensor–tensor ──────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_add")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Add    (IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_sub")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Sub    (IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_mul")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Mul    (IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_div")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Div    (IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_pow")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Pow    (IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_maximum")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Maximum(IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_minimum")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Minimum(IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);

    // ── Scalar binary ─────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_add_scalar")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int AddScalar(IntPtr exec, IntPtr dst, IntPtr a, double s);
    [LibraryImport(LibName, EntryPoint = "matx_sub_scalar")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int SubScalar(IntPtr exec, IntPtr dst, IntPtr a, double s);
    [LibraryImport(LibName, EntryPoint = "matx_mul_scalar")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int MulScalar(IntPtr exec, IntPtr dst, IntPtr a, double s);
    [LibraryImport(LibName, EntryPoint = "matx_div_scalar")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int DivScalar(IntPtr exec, IntPtr dst, IntPtr a, double s);
    [LibraryImport(LibName, EntryPoint = "matx_pow_scalar")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int PowScalar(IntPtr exec, IntPtr dst, IntPtr a, double s);

    // ── Reductions ────────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_sum")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Sum  (IntPtr exec, IntPtr dst, IntPtr src, int axis);
    [LibraryImport(LibName, EntryPoint = "matx_mean")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Mean (IntPtr exec, IntPtr dst, IntPtr src, int axis);
    [LibraryImport(LibName, EntryPoint = "matx_max_r")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int MaxR (IntPtr exec, IntPtr dst, IntPtr src, int axis);
    [LibraryImport(LibName, EntryPoint = "matx_min_r")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int MinR (IntPtr exec, IntPtr dst, IntPtr src, int axis);
    [LibraryImport(LibName, EntryPoint = "matx_prod")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Prod (IntPtr exec, IntPtr dst, IntPtr src, int axis);
    [LibraryImport(LibName, EntryPoint = "matx_var")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Var  (IntPtr exec, IntPtr dst, IntPtr src, int axis);
    [LibraryImport(LibName, EntryPoint = "matx_stdd")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Stdd (IntPtr exec, IntPtr dst, IntPtr src, int axis);
    [LibraryImport(LibName, EntryPoint = "matx_any")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Any  (IntPtr exec, IntPtr dst, IntPtr src);
    [LibraryImport(LibName, EntryPoint = "matx_norm")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Norm (IntPtr exec, IntPtr dst, IntPtr src);
    [LibraryImport(LibName, EntryPoint = "matx_trace")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Trace(IntPtr exec, IntPtr dst, IntPtr src);
    [LibraryImport(LibName, EntryPoint = "matx_cumsum")][UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int CumSum(IntPtr exec, IntPtr dst, IntPtr src, int axis);
    [LibraryImport(LibName, EntryPoint = "matx_sort")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Sort (IntPtr exec, IntPtr dst, IntPtr src, SortDir dir);

    // ── Generators ────────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_fill_zeros")]          [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillZeros        (IntPtr exec, IntPtr dst);
    [LibraryImport(LibName, EntryPoint = "matx_fill_ones")]           [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillOnes         (IntPtr exec, IntPtr dst);
    [LibraryImport(LibName, EntryPoint = "matx_fill_random_uniform")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillRandomUniform(IntPtr exec, IntPtr dst);
    [LibraryImport(LibName, EntryPoint = "matx_fill_random_normal")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillRandomNormal (IntPtr exec, IntPtr dst);
    [LibraryImport(LibName, EntryPoint = "matx_fill_linspace")]       [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillLinspace      (IntPtr exec, IntPtr dst, double start, double stop);
    [LibraryImport(LibName, EntryPoint = "matx_fill_range")]          [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillRange         (IntPtr exec, IntPtr dst, double start, double step);
    [LibraryImport(LibName, EntryPoint = "matx_fill_fftfreq")]        [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillFftFreq       (IntPtr exec, IntPtr dst, double sampleRate);
    [LibraryImport(LibName, EntryPoint = "matx_fill_hamming")]        [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillHamming       (IntPtr exec, IntPtr dst);
    [LibraryImport(LibName, EntryPoint = "matx_fill_hanning")]        [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillHanning       (IntPtr exec, IntPtr dst);
    [LibraryImport(LibName, EntryPoint = "matx_fill_blackman")]       [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillBlackman      (IntPtr exec, IntPtr dst);
    [LibraryImport(LibName, EntryPoint = "matx_fill_bartlett")]       [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FillBartlett      (IntPtr exec, IntPtr dst);

    // ── FFT ───────────────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_fft")]      [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Fft     (IntPtr exec, IntPtr dst, IntPtr src, long nfft, FftNorm norm);
    [LibraryImport(LibName, EntryPoint = "matx_ifft")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Ifft    (IntPtr exec, IntPtr dst, IntPtr src, long nfft, FftNorm norm);
    [LibraryImport(LibName, EntryPoint = "matx_fft2")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Fft2    (IntPtr exec, IntPtr dst, IntPtr src, FftNorm norm);
    [LibraryImport(LibName, EntryPoint = "matx_ifft2")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Ifft2   (IntPtr exec, IntPtr dst, IntPtr src, FftNorm norm);
    [LibraryImport(LibName, EntryPoint = "matx_rfft")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Rfft    (IntPtr exec, IntPtr dst, IntPtr src, long nfft, FftNorm norm);
    [LibraryImport(LibName, EntryPoint = "matx_irfft")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Irfft   (IntPtr exec, IntPtr dst, IntPtr src, long nfft, FftNorm norm);
    [LibraryImport(LibName, EntryPoint = "matx_fftshift")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int FftShift (IntPtr exec, IntPtr dst, IntPtr src);
    [LibraryImport(LibName, EntryPoint = "matx_ifftshift")][UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int IFftShift(IntPtr exec, IntPtr dst, IntPtr src);
    [LibraryImport(LibName, EntryPoint = "matx_dct")]      [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Dct     (IntPtr exec, IntPtr dst, IntPtr src);

    // ── Linear algebra ────────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_matmul")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Matmul   (IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_matvec")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Matvec   (IntPtr exec, IntPtr dst, IntPtr a, IntPtr x);
    [LibraryImport(LibName, EntryPoint = "matx_outer")]     [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Outer    (IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_transpose")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Transpose (IntPtr exec, IntPtr dst, IntPtr src);
    [LibraryImport(LibName, EntryPoint = "matx_hermitian")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Hermitian (IntPtr exec, IntPtr dst, IntPtr src);

    [LibraryImport(LibName, EntryPoint = "matx_svd")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Svd  (IntPtr exec, IntPtr u, IntPtr s, IntPtr vt, IntPtr a, SvdMode mode);
    [LibraryImport(LibName, EntryPoint = "matx_svdpi")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int SvdPi(IntPtr exec, IntPtr u, IntPtr s, IntPtr vt, IntPtr a, int iters, int k);
    [LibraryImport(LibName, EntryPoint = "matx_qr")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Qr   (IntPtr exec, IntPtr q, IntPtr r, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_lu")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Lu   (IntPtr exec, IntPtr l, IntPtr u, IntPtr piv, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_chol")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Chol (IntPtr exec, IntPtr l, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_eig")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Eig  (IntPtr exec, IntPtr vec, IntPtr val, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_solve")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Solve(IntPtr exec, IntPtr x, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_inv")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Inv  (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_pinv")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Pinv (IntPtr exec, IntPtr dst, IntPtr a);
    [LibraryImport(LibName, EntryPoint = "matx_det")]   [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Det  (IntPtr exec, IntPtr dst, IntPtr a);

    [LibraryImport(LibName, EntryPoint = "matx_einsum2", StringMarshalling = StringMarshalling.Utf8)]
    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    internal static partial int Einsum2(IntPtr exec, IntPtr dst, string subscripts, IntPtr a, IntPtr b);

    // ── Signal processing ─────────────────────────────────────────────────────
    [LibraryImport(LibName, EntryPoint = "matx_conv1d")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Conv1d      (IntPtr exec, IntPtr dst, IntPtr sig, IntPtr filt, ConvMode mode);
    [LibraryImport(LibName, EntryPoint = "matx_conv2d")]  [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Conv2d      (IntPtr exec, IntPtr dst, IntPtr sig, IntPtr filt, ConvMode mode);
    [LibraryImport(LibName, EntryPoint = "matx_corr")]    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Corr        (IntPtr exec, IntPtr dst, IntPtr a, IntPtr b);
    [LibraryImport(LibName, EntryPoint = "matx_pwelch")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int Pwelch      (IntPtr exec, IntPtr pxx, IntPtr sig, IntPtr win, long noverlap, long nfft);
    [LibraryImport(LibName, EntryPoint = "matx_resample_poly")] [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])] internal static partial int ResamplePoly(IntPtr exec, IntPtr dst, IntPtr src, int up, int down, IntPtr filter);
}
