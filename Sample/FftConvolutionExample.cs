// FftConvolutionExample.cs
// Demonstrates FFT-based convolution:
//   1. Generate a random float32 signal.
//   2. Apply a Hamming window of same length as FIR filter.
//   3. Compute FFT of signal and filter, multiply in frequency domain, IFFT.
// Mirrors examples/fft_conv.cu.
namespace MatX.Net.Examples;

public static class FftConvolutionExample
{
    public static void Run()
    {
        const int N      = 1024;   // signal length
        const int FilterN = 64;    // FIR filter length
        const int Nfft   = N + FilterN - 1; // full linear convolution size

        using var exec = Executor.CreateCuda();

        // ── Allocate ────────────────────────────────────────────────────────
        using var signal   = Tensor.CreateFloat32(  [N]);
        using var filter   = Tensor.CreateFloat32(  [FilterN]);
        using var window   = Tensor.CreateFloat32(  [FilterN]);
        using var sigFreq  = Tensor.CreateComplex64([Nfft]);
        using var filtFreq = Tensor.CreateComplex64([Nfft]);
        using var outFreq  = Tensor.CreateComplex64([Nfft]);
        using var output   = Tensor.CreateComplex64([Nfft]);

        signal.Name  = "signal";
        filter.Name  = "filter";
        output.Name  = "output";

        // ── Initialise ──────────────────────────────────────────────────────
        Ops.RandomNormal (exec, signal);   // random input
        Ops.Hamming      (exec, window);   // Hamming window (same size as filter)
        Ops.RandomUniform(exec, filter);

        // Apply window to filter  (filter *= window, stored in filter)
        Ops.Mul(exec, filter, filter, window);

        // ── FFT convolution ──────────────────────────────────────────────────
        // Zero-padded FFT of signal and filter
        Ops.Rfft(exec, sigFreq,  signal, nfft: Nfft);
        Ops.Rfft(exec, filtFreq, filter, nfft: Nfft);

        // Multiply in frequency domain
        Ops.Mul(exec, outFreq, sigFreq, filtFreq);

        // Inverse FFT
        Ops.Irfft(exec, output, outFreq, nfft: Nfft);

        exec.Synchronize();

        // ── Verify ──────────────────────────────────────────────────────────
        // Compute energy of output via sum(abs2(output))
        using var energy = Tensor.CreateFloat32([1]);
        using var absOut = Tensor.CreateFloat32([Nfft]);
        Ops.Abs2(exec, absOut, output);
        Ops.Sum (exec, energy, absOut);
        exec.Synchronize();

        float e = energy.ToArray<float>()[0];
        Console.WriteLine($"  Signal length   : {N}");
        Console.WriteLine($"  Filter length   : {FilterN}");
        Console.WriteLine($"  Output length   : {Nfft}");
        Console.WriteLine($"  Output energy   : {e:F4}");
        Console.WriteLine("  [OK]");
    }
}
