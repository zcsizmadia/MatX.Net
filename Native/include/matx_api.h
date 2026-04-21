// matx_api.h  –  Cross-platform C API for the MatX CUDA tensor library
// Suitable for P/Invoke from .NET (Linux, Windows; macOS host-only).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// ── DLL visibility ───────────────────────────────────────────────────────────
#if defined(_WIN32)
#  if defined(MATX_NATIVE_EXPORTS)
#    define MATX_API __declspec(dllexport)
#  else
#    define MATX_API __declspec(dllimport)
#  endif
#else
#  define MATX_API __attribute__((visibility("default")))
#endif

// ── Opaque handles ───────────────────────────────────────────────────────────
typedef void* MatxTensorHandle;
typedef void* MatxExecutorHandle;

// ── Enumerations ─────────────────────────────────────────────────────────────
typedef enum {
    MATX_DTYPE_FLOAT32    = 0,
    MATX_DTYPE_FLOAT64    = 1,
    MATX_DTYPE_COMPLEX64  = 2,   ///< cuda::std::complex<float>
    MATX_DTYPE_COMPLEX128 = 3,   ///< cuda::std::complex<double>
    MATX_DTYPE_INT32      = 4,
    MATX_DTYPE_INT64      = 5,
    MATX_DTYPE_UINT32     = 6,
    MATX_DTYPE_UINT8      = 7,
} MatxDtype;

typedef enum {
    MATX_MEMORY_MANAGED     = 0,  ///< cudaMallocManaged (default – host + device)
    MATX_MEMORY_DEVICE      = 1,  ///< cudaMalloc (device only)
    MATX_MEMORY_HOST        = 2,  ///< cudaHostAlloc (pinned)
    MATX_MEMORY_HOST_MALLOC = 3,  ///< malloc (pageable, no GPU DMA)
} MatxMemorySpace;

typedef enum {
    MATX_CONV_FULL  = 0,
    MATX_CONV_SAME  = 1,
    MATX_CONV_VALID = 2,
} MatxConvMode;

typedef enum {
    MATX_FFT_NORM_BACKWARD = 0,  ///< no normalization on forward, 1/N on inverse
    MATX_FFT_NORM_FORWARD  = 1,  ///< 1/N on forward, no normalization on inverse
    MATX_FFT_NORM_ORTHO    = 2,  ///< 1/sqrt(N) on both
} MatxFftNorm;

typedef enum {
    MATX_SORT_ASC  = 0,
    MATX_SORT_DESC = 1,
} MatxSortDir;

typedef enum {
    MATX_SVD_REDUCED = 0,
    MATX_SVD_FULL    = 1,
    MATX_SVD_NONE    = 2,
} MatxSvdMode;

// ── Error codes ──────────────────────────────────────────────────────────────
#define MATX_OK              0
#define MATX_ERR_ARGS        1
#define MATX_ERR_CUDA        2
#define MATX_ERR_UNSUPPORTED 3
#define MATX_ERR_TYPE        4   ///< tensor type or rank mismatch
#define MATX_ERR_INTERNAL    5

// ── Error query ──────────────────────────────────────────────────────────────
MATX_API const char* matx_get_last_error(void);
MATX_API void        matx_clear_error(void);

// ─────────────────────────────────────────────────────────────────────────────
//  Tensor life-cycle
// ─────────────────────────────────────────────────────────────────────────────

/// Create an owning tensor.  shape[] has `rank` elements.
MATX_API MatxTensorHandle matx_tensor_create(
    MatxDtype         dtype,
    int               rank,
    const int64_t*    shape,
    MatxMemorySpace   memory_space);

/// Wrap an existing raw pointer as a non-owning view.
/// The pointer must remain valid for the tensor's lifetime.
MATX_API MatxTensorHandle matx_tensor_from_ptr(
    MatxDtype       dtype,
    int             rank,
    const int64_t*  shape,
    void*           data);

MATX_API void matx_tensor_destroy(MatxTensorHandle handle);

// ─────────────────────────────────────────────────────────────────────────────
//  Tensor metadata
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int        matx_tensor_rank      (MatxTensorHandle h);
MATX_API MatxDtype  matx_tensor_dtype     (MatxTensorHandle h);
MATX_API int64_t    matx_tensor_size      (MatxTensorHandle h, int dim);
MATX_API int64_t    matx_tensor_stride    (MatxTensorHandle h, int dim);
MATX_API int64_t    matx_tensor_total_size(MatxTensorHandle h);
MATX_API void*      matx_tensor_data      (MatxTensorHandle h);
MATX_API void       matx_tensor_set_name  (MatxTensorHandle h, const char* name);
MATX_API void       matx_tensor_print     (MatxTensorHandle h);

// ─────────────────────────────────────────────────────────────────────────────
//  Executor life-cycle
// ─────────────────────────────────────────────────────────────────────────────

/// stream == nullptr  →  default CUDA stream.
MATX_API MatxExecutorHandle matx_executor_cuda(void* stream);
MATX_API MatxExecutorHandle matx_executor_host(void);
MATX_API void               matx_executor_destroy(MatxExecutorHandle h);
MATX_API int                matx_executor_sync   (MatxExecutorHandle h);

/// Returns the underlying cudaStream_t (null for host executors).
MATX_API void*              matx_executor_stream (MatxExecutorHandle h);

// ─────────────────────────────────────────────────────────────────────────────
//  Memory helpers
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int matx_tensor_prefetch_device(MatxTensorHandle h, MatxExecutorHandle exec);
MATX_API int matx_tensor_copy(MatxTensorHandle dst, MatxTensorHandle src, MatxExecutorHandle exec);

// ─────────────────────────────────────────────────────────────────────────────
//  View / shape operations  (return new handle sharing the same data buffer)
// ─────────────────────────────────────────────────────────────────────────────

/// Reshape a contiguous tensor.  Total element count must be preserved.
MATX_API MatxTensorHandle matx_tensor_reshape(
    MatxTensorHandle h,
    int              new_rank,
    const int64_t*   new_shape);

/// Transpose axes.  axes[] contains a permutation of [0..rank-1].
MATX_API MatxTensorHandle matx_tensor_permute(
    MatxTensorHandle h,
    const int32_t*   axes);

/// Slice (sub-tensor view).
/// Use INT64_MIN for matxKeepDim / INT64_MAX for matxEnd.
MATX_API MatxTensorHandle matx_tensor_slice(
    MatxTensorHandle h,
    const int64_t*   starts,
    const int64_t*   ends,
    const int64_t*   strides);  ///< nullptr → all ones

/// Collapse all dimensions into a 1-D view.
MATX_API MatxTensorHandle matx_tensor_flatten(MatxTensorHandle h);

// ─────────────────────────────────────────────────────────────────────────────
//  Element-wise unary operations   dst = f(a)
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int matx_abs    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_abs2   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_sqrt   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_exp    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_log    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_log2   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_log10  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_sin    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_cos    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_tan    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_asin   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_acos   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_atan   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_sinh   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_cosh   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_tanh   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_ceil   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_floor  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_round  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_sign   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_neg    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_conj   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_real   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_imag   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_normcdf(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_erf    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_erfc   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);

// ─────────────────────────────────────────────────────────────────────────────
//  Element-wise binary tensor–tensor operations   dst = f(a, b)
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int matx_add     (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_sub     (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_mul     (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_div     (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_pow     (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_maximum (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_minimum (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);

// ── Scalar variants (double covers float/int through safe cast) ───────────────
MATX_API int matx_add_scalar(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, double s);
MATX_API int matx_sub_scalar(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, double s);
MATX_API int matx_mul_scalar(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, double s);
MATX_API int matx_div_scalar(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, double s);
MATX_API int matx_pow_scalar(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, double s);

// ─────────────────────────────────────────────────────────────────────────────
//  Reduction operations
//  axis == -1  →  reduce all dimensions (dst is a rank-0 or rank-1 scalar tensor)
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int matx_sum   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int axis);
MATX_API int matx_mean  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int axis);
MATX_API int matx_max_r (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int axis);
MATX_API int matx_min_r (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int axis);
MATX_API int matx_prod  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int axis);
MATX_API int matx_var   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int axis);
MATX_API int matx_stdd  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int axis);
MATX_API int matx_any   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src);
MATX_API int matx_norm  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src);
MATX_API int matx_trace (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src);
MATX_API int matx_cumsum(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int axis);
MATX_API int matx_sort  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, MatxSortDir dir);

// ─────────────────────────────────────────────────────────────────────────────
//  Generators  (fill an existing tensor in-place)
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int matx_fill_zeros         (MatxExecutorHandle exec, MatxTensorHandle dst);
MATX_API int matx_fill_ones          (MatxExecutorHandle exec, MatxTensorHandle dst);
MATX_API int matx_fill_random_uniform(MatxExecutorHandle exec, MatxTensorHandle dst);
MATX_API int matx_fill_random_normal (MatxExecutorHandle exec, MatxTensorHandle dst);
MATX_API int matx_fill_linspace      (MatxExecutorHandle exec, MatxTensorHandle dst, double start, double stop);
MATX_API int matx_fill_range         (MatxExecutorHandle exec, MatxTensorHandle dst, double start, double step);
MATX_API int matx_fill_fftfreq       (MatxExecutorHandle exec, MatxTensorHandle dst, double sample_rate);
MATX_API int matx_fill_hamming       (MatxExecutorHandle exec, MatxTensorHandle dst);
MATX_API int matx_fill_hanning       (MatxExecutorHandle exec, MatxTensorHandle dst);
MATX_API int matx_fill_blackman      (MatxExecutorHandle exec, MatxTensorHandle dst);
MATX_API int matx_fill_bartlett      (MatxExecutorHandle exec, MatxTensorHandle dst);

// ─────────────────────────────────────────────────────────────────────────────
//  FFT  (nfft == 0 → use source size)
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int matx_fft    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int64_t nfft, MatxFftNorm norm);
MATX_API int matx_ifft   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int64_t nfft, MatxFftNorm norm);
MATX_API int matx_fft2   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, MatxFftNorm norm);
MATX_API int matx_ifft2  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, MatxFftNorm norm);
MATX_API int matx_rfft   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int64_t nfft, MatxFftNorm norm);
MATX_API int matx_irfft  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src, int64_t nfft, MatxFftNorm norm);
MATX_API int matx_fftshift (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src);
MATX_API int matx_ifftshift(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src);
MATX_API int matx_dct    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src);

// ─────────────────────────────────────────────────────────────────────────────
//  Linear algebra
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int matx_matmul   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_matvec   (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle x);
MATX_API int matx_outer    (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_transpose(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src);
MATX_API int matx_hermitian(MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle src);

// Decompositions – all output tensors must be pre-allocated to the correct shape.
MATX_API int matx_svd  (MatxExecutorHandle exec,
                         MatxTensorHandle u, MatxTensorHandle s, MatxTensorHandle vt,
                         MatxTensorHandle a, MatxSvdMode mode);
MATX_API int matx_svdpi(MatxExecutorHandle exec,
                         MatxTensorHandle u, MatxTensorHandle s, MatxTensorHandle vt,
                         MatxTensorHandle a, int iterations, int k);
MATX_API int matx_qr   (MatxExecutorHandle exec,
                         MatxTensorHandle q, MatxTensorHandle r,
                         MatxTensorHandle a);
MATX_API int matx_lu   (MatxExecutorHandle exec,
                         MatxTensorHandle l, MatxTensorHandle u_out, MatxTensorHandle piv,
                         MatxTensorHandle a);
MATX_API int matx_chol (MatxExecutorHandle exec,
                         MatxTensorHandle l,
                         MatxTensorHandle a);
MATX_API int matx_eig  (MatxExecutorHandle exec,
                         MatxTensorHandle vectors, MatxTensorHandle values,
                         MatxTensorHandle a);
MATX_API int matx_solve(MatxExecutorHandle exec,
                         MatxTensorHandle x,
                         MatxTensorHandle a, MatxTensorHandle b);
MATX_API int matx_inv  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_pinv (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);
MATX_API int matx_det  (MatxExecutorHandle exec, MatxTensorHandle dst, MatxTensorHandle a);

/// Einstein summation (cuTENSOR backend).
MATX_API int matx_einsum2(MatxExecutorHandle exec, MatxTensorHandle dst,
                           const char* subscripts,
                           MatxTensorHandle a, MatxTensorHandle b);

// ─────────────────────────────────────────────────────────────────────────────
//  Signal processing
// ─────────────────────────────────────────────────────────────────────────────
MATX_API int matx_conv1d(MatxExecutorHandle exec,
                          MatxTensorHandle dst,
                          MatxTensorHandle signal, MatxTensorHandle filter,
                          MatxConvMode mode);
MATX_API int matx_conv2d(MatxExecutorHandle exec,
                          MatxTensorHandle dst,
                          MatxTensorHandle signal, MatxTensorHandle filter,
                          MatxConvMode mode);
MATX_API int matx_corr  (MatxExecutorHandle exec,
                          MatxTensorHandle dst,
                          MatxTensorHandle a, MatxTensorHandle b);

/// Welch PSD.  window must be a 1-D float tensor.
MATX_API int matx_pwelch(MatxExecutorHandle exec,
                          MatxTensorHandle pxx,
                          MatxTensorHandle signal,
                          MatxTensorHandle window,
                          int64_t noverlap,
                          int64_t nfft);

MATX_API int matx_resample_poly(MatxExecutorHandle exec,
                                 MatxTensorHandle dst,
                                 MatxTensorHandle src,
                                 int up, int down,
                                 MatxTensorHandle filter);   ///< nullptr → default

#ifdef __cplusplus
}
#endif
