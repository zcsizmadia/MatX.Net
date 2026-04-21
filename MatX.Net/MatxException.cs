// MatxException.cs – Exception type carrying the native error string.
namespace MatX.Net;

/// <summary>
/// Thrown when a native MatX operation returns a non-zero error code.
/// </summary>
public sealed class MatxException : Exception
{
    public int ErrorCode { get; }

    internal MatxException(int code, string message)
        : base(message)
    {
        ErrorCode = code;
    }
}
