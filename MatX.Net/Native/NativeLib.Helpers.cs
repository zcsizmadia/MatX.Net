// NativeHelpers.cs – Internal helper to check return codes.
using System.Runtime.InteropServices;
using MatX.Net.Native;

namespace MatX.Net.Native;

internal static partial class NativeLib
{
    /// <summary>Throw <see cref="MatxException"/> if <paramref name="code"/> is non-zero.</summary>
    internal static void Check(int code)
    {
        if (code == 0) return;
        string msg = Marshal.PtrToStringUTF8(NativeLib.GetLastError()) ?? "unknown error";
        NativeLib.ClearError();
        throw new MatxException(code, msg);
    }

    /// <summary>
    /// Throw <see cref="MatxException"/> if <paramref name="handle"/> is zero/null.
    /// </summary>
    internal static IntPtr CheckHandle(IntPtr handle)
    {
        if (handle != IntPtr.Zero) return handle;
        string msg = Marshal.PtrToStringUTF8(NativeLib.GetLastError()) ?? "native returned null handle";
        NativeLib.ClearError();
        throw new MatxException(-1, msg);
    }
}
