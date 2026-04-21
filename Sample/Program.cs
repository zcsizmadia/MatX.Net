// Program.cs – Entry point that runs all examples.
using MatX.Net.Examples;

Console.WriteLine("=== MatX C# Wrapper Examples ===\n");

var examples = new (string Name, Action Run)[]
{
    ("01 – Convolution (FFT method)",           FftConvolutionExample.Run),
    ("02 – SVD Power Iteration",                SvdPowerExample.Run),
    ("03 – Black-Scholes Option Pricing",       BlackScholesExample.Run),
    ("04 – Spectrogram (STFT / Pwelch)",        SpectrogramExample.Run),
    ("05 – Matrix Decompositions (QR/LU/Chol)", DecompositionsExample.Run),
};

foreach (var (name, run) in examples)
{
    Console.WriteLine($"──────────────────────────────────────────");
    Console.WriteLine($"  {name}");
    Console.WriteLine($"──────────────────────────────────────────");
    try
    {
        run();
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  [FAILED] {ex.Message}");
    }
    Console.WriteLine();
}

Console.WriteLine("Done.");
