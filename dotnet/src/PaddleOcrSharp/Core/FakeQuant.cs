namespace PaddleOcrSharp.Core;

/// <summary>
/// Experiment hook: emulates quantising a product's activations to a signed integer width, one scale
/// per row (per token), and nothing else — the arithmetic stays float32.
/// </summary>
/// <remarks>
/// Set <c>PADDLEOCR_SHARP_FAKE_A_BITS=8</c> to round every activation row that feeds a checkpoint
/// weight to int8 and back before the product. It exists to measure what an integer-activation
/// kernel would cost in accuracy before anyone writes one; it is not a code path to ship.
/// </remarks>
internal static class FakeQuant
{
    /// <summary>The emulated activation width in bits, or 0 for off.</summary>
    internal static readonly int ActivationBits =
        int.TryParse(Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_FAKE_A_BITS"), out int bits) ? bits : 0;

    /// <summary>
    /// Products with fewer rows than this are left alone (default 32), because an integer-activation
    /// kernel would only be used where the product is bound by arithmetic: the tower, the projector
    /// and the prompt's prefill. A decode step is bound by bytes, and stays weight-only.
    /// </summary>
    internal static readonly int MinRows =
        int.TryParse(Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_FAKE_A_MINROWS"), out int rows) ? rows : 32;

    /// <summary>Returns a quantised copy of <paramref name="x"/>, row by row.</summary>
    internal static PooledBuffer Activations(ReadOnlySpan<float> x, int rows, int inner)
    {
        PooledBuffer copy = TensorPool.Rent(rows * inner);
        Span<float> destination = copy.Span;
        float levels = (1 << (ActivationBits - 1)) - 1;

        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> source = x.Slice(r * inner, inner);
            Span<float> target = destination.Slice(r * inner, inner);

            float peak = 0f;
            foreach (float value in source)
            {
                peak = Math.Max(peak, Math.Abs(value));
            }

            float scale = peak == 0f ? 1f : peak / levels;
            for (int i = 0; i < inner; i++)
            {
                target[i] = Math.Clamp(MathF.Round(source[i] / scale), -levels, levels) * scale;
            }
        }

        return copy;
    }
}
