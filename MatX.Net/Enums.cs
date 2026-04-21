// Enums.cs – C# mirrors of the C API enumerations.
namespace MatX.Net;

/// <summary>Element data type.</summary>
public enum DType : int
{
    Float32    = 0,
    Float64    = 1,
    Complex64  = 2,
    Complex128 = 3,
    Int32      = 4,
    Int64      = 5,
    UInt32     = 6,
    UInt8      = 7,
}

/// <summary>Device memory space for tensor allocation.</summary>
public enum MemorySpace : int
{
    /// <summary>cudaMallocManaged – accessible from both host and device (default).</summary>
    Managed     = 0,
    /// <summary>cudaMalloc – device only.</summary>
    Device      = 1,
    /// <summary>cudaHostAlloc – pinned host memory.</summary>
    Host        = 2,
    /// <summary>malloc – pageable host memory.</summary>
    HostMalloc  = 3,
}

/// <summary>Convolution output size convention.</summary>
public enum ConvMode : int
{
    Full  = 0,
    Same  = 1,
    Valid = 2,
}

/// <summary>FFT normalization convention.</summary>
public enum FftNorm : int
{
    /// <summary>No normalization on forward, 1/N on inverse.</summary>
    Backward = 0,
    /// <summary>1/N on forward, no normalization on inverse.</summary>
    Forward  = 1,
    /// <summary>1/√N on both.</summary>
    Ortho    = 2,
}

/// <summary>Sort direction.</summary>
public enum SortDir : int
{
    Ascending  = 0,
    Descending = 1,
}

/// <summary>SVD mode.</summary>
public enum SvdMode : int
{
    Reduced = 0,
    Full    = 1,
    None    = 2,
}
