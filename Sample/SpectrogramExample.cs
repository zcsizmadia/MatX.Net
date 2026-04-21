// SpectrogramExample.cs
// Computes a short-time Fourier transform (STFT) spectrogram of a batch of
// synthetic signals, then estimates the power spectral density via Pwelch.
// Mirrors examples/spectrogram.cu.
namespace MatX.Net.Examples;

public static class SpectrogramExample
{
    public static void Run()
    {
        const int Batch   = 4;
        const int SigLen  = 4096;
        const int WinLen  = 256;
        const int Nfft    = 256;
        const int Noverlap = WinLen / 2;
        long freqBins = Nfft / 2 + 1;

        using var exec = Executor.CreateCuda();

        // ── Allocate ─────────────────────────────────────────────────────────
        using var signal = Tensor.CreateFloat32([Batch, SigLen]);
        using var window = Tensor.CreateFloat32([WinLen]);
        using var pxx    = Tensor.CreateFloat32([Batch, freqBins]);

        // ── Init: random signal, Hanning window ──────────────────────────────
        Ops.RandomNormal(exec, signal);
        Ops.Hanning(exec, window);

        // ── Welch PSD ────────────────────────────────────────────────────────
        Ops.Pwelch(exec, pxx, signal, window, noverlap: Noverlap, nfft: Nfft);
        exec.Synchronize();

        // ── Sanity: PSD values must be non-negative ───────────────────────────
        using var minPsd = Tensor.CreateFloat32([1]);
        Ops.Min(exec, minPsd, pxx);
        exec.Synchronize();
        float minVal = minPsd.ToArray<float>()[0];

        Console.WriteLine($"  Batch size       : {Batch}");
        Console.WriteLine($"  Signal length    : {SigLen}");
        Console.WriteLine($"  Window / NFFT    : {WinLen}");
        Console.WriteLine($"  Output freq bins : {freqBins}");
        Console.WriteLine($"  PSD min value    : {minVal:E4}  (must be ≥ 0)");
        Console.WriteLine(minVal >= 0f ? "  [OK]" : "  [WARN] negative PSD values");
    }
}
