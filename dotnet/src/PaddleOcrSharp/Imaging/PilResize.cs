using System.Buffers;

using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Imaging;

/// <summary>
/// A byte-exact port of Pillow's <c>ImagingResample</c> for 8-bit images.
/// </summary>
/// <remarks>
/// <para>
/// The upstream preprocessor calls <c>transformers.image_transforms.resize</c>, which converts
/// the uint8 array to a <c>PIL.Image</c> and calls <c>Image.resize(..., BICUBIC)</c>. Any other
/// bicubic implementation — including SkiaSharp's — differs in the filter support scaling used
/// when downsampling, in the fixed-point coefficient rounding, and in the uint8 rounding between
/// the horizontal and vertical passes. Those differences are large enough to move OCR output, so
/// this reproduces Pillow's algorithm exactly:
/// </para>
/// <list type="bullet">
///   <item>filter support is scaled by <c>max(1, in/out)</c>, giving antialiasing on downscale;</item>
///   <item>coefficients are quantised to <c>Q22</c> with round-half-away-from-zero;</item>
///   <item>accumulation is integer with a <c>1 &lt;&lt; 21</c> bias, then an arithmetic shift and a saturating clip;</item>
///   <item>the horizontal pass runs first and its uint8 result feeds the vertical pass.</item>
/// </list>
/// <para>Reference: <c>Pillow/src/libImaging/Resample.c</c>.</para>
/// </remarks>
public static class PilResize
{
    /// <summary>Fixed-point fraction bits, Pillow's <c>PRECISION_BITS</c>.</summary>
    private const int PrecisionBits = 32 - 8 - 2;

    /// <summary>Rounding bias added before the final arithmetic shift.</summary>
    private const int RoundingBias = 1 << (PrecisionBits - 1);

    /// <summary>Catmull-Rom <c>a</c> parameter; Pillow hard-codes −0.5.</summary>
    private const double BicubicA = -0.5;

    /// <summary>Lobe count of Pillow's Lanczos filter.</summary>
    private const double LanczosLobes = 3.0;

    /// <summary>Resampling filters this port implements.</summary>
    public enum Filter
    {
        /// <summary>Pillow's <c>BICUBIC</c>: Catmull-Rom with <c>a = −0.5</c>, support 2.</summary>
        Bicubic,

        /// <summary>Pillow's <c>LANCZOS</c>: a three-lobe windowed sinc, support 3.</summary>
        Lanczos,
    }

    /// <summary>
    /// Resizes <paramref name="source"/> to <paramref name="width"/> × <paramref name="height"/>
    /// with Pillow's bicubic filter.
    /// </summary>
    public static RgbImage ResizeBicubic(RgbImage source, int width, int height) =>
        Resize(source, width, height, Filter.Bicubic);

    /// <summary>
    /// Resizes <paramref name="source"/> to <paramref name="width"/> × <paramref name="height"/>
    /// with Pillow's Lanczos filter, which is what the spotting pre-process uses.
    /// </summary>
    public static RgbImage ResizeLanczos(RgbImage source, int width, int height) =>
        Resize(source, width, height, Filter.Lanczos);

    /// <summary>
    /// Resizes <paramref name="source"/> to <paramref name="width"/> × <paramref name="height"/>.
    /// </summary>
    public static RgbImage Resize(RgbImage source, int width, int height, Filter filter)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);

        bool needHorizontal = width != source.Width;
        bool needVertical = height != source.Height;

        if (!needHorizontal && !needVertical)
        {
            return source.Clone();
        }

        RgbImage current = source;
        RgbImage? intermediate = null;

        if (needHorizontal)
        {
            intermediate = RgbImage.Rent(width, source.Height);
            ResampleHorizontal(source, intermediate, filter);
            current = intermediate;
        }

        if (!needVertical)
        {
            return current;
        }

        RgbImage result = RgbImage.Rent(width, height);
        ResampleVertical(current, result, filter);
        intermediate?.Dispose();
        return result;
    }

    /// <summary>
    /// Precomputes Pillow's per-output-pixel filter taps.
    /// </summary>
    /// <param name="inSize">Source extent along the axis.</param>
    /// <param name="outSize">Destination extent along the axis.</param>
    /// <param name="filter">Which reconstruction filter to sample.</param>
    /// <param name="bounds">Receives <c>(min, count)</c> pairs, two ints per output pixel.</param>
    /// <param name="coefficients">Receives <c>outSize × kernelSize</c> Q22 coefficients.</param>
    /// <returns>The kernel size (taps per output pixel).</returns>
    internal static int PrecomputeCoefficients(
        int inSize,
        int outSize,
        Filter filter,
        out int[] bounds,
        out int[] coefficients)
    {
        double scale = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        double support = (filter == Filter.Lanczos ? LanczosLobes : 2.0) * filterScale;

        int kernelSize = ((int)Math.Ceiling(support) * 2) + 1;

        bounds = new int[outSize * 2];
        coefficients = new int[outSize * kernelSize];
        double[] taps = ArrayPool<double>.Shared.Rent(kernelSize);

        try
        {
            for (int index = 0; index < outSize; index++)
            {
                double center = (index + 0.5) * scale;
                double inverseScale = 1.0 / filterScale;

                int min = (int)(center - support + 0.5);
                if (min < 0)
                {
                    min = 0;
                }

                int max = (int)(center + support + 0.5);
                if (max > inSize)
                {
                    max = inSize;
                }

                int count = max - min;

                double weightSum = 0.0;
                for (int k = 0; k < count; k++)
                {
                    double w = Evaluate(filter, (k + min - center + 0.5) * inverseScale);
                    taps[k] = w;
                    weightSum += w;
                }

                int baseOffset = index * kernelSize;
                for (int k = 0; k < count; k++)
                {
                    double w = weightSum != 0.0 ? taps[k] / weightSum : taps[k];
                    coefficients[baseOffset + k] = w < 0
                        ? (int)(-0.5 + (w * (1 << PrecisionBits)))
                        : (int)(0.5 + (w * (1 << PrecisionBits)));
                }

                bounds[(index * 2) + 0] = min;
                bounds[(index * 2) + 1] = count;
            }
        }
        finally
        {
            ArrayPool<double>.Shared.Return(taps);
        }

        return kernelSize;
    }

    private static void ResampleHorizontal(RgbImage source, RgbImage destination, Filter filter)
    {
        int kernelSize = PrecomputeCoefficients(
            source.Width, destination.Width, filter, out int[] bounds, out int[] coefficients);

        int height = source.Height;
        int outWidth = destination.Width;

        Parallelism.For(height, y =>
        {
            ReadOnlySpan<byte> srcRow = source.Row(y);
            Span<byte> dstRow = destination.Row(y);

            for (int x = 0; x < outWidth; x++)
            {
                int min = bounds[(x * 2) + 0];
                int count = bounds[(x * 2) + 1];
                int taps = x * kernelSize;

                int r = RoundingBias;
                int g = RoundingBias;
                int b = RoundingBias;

                for (int k = 0; k < count; k++)
                {
                    int weight = coefficients[taps + k];
                    int offset = (min + k) * 3;
                    r += srcRow[offset] * weight;
                    g += srcRow[offset + 1] * weight;
                    b += srcRow[offset + 2] * weight;
                }

                int outOffset = x * 3;
                dstRow[outOffset] = Clip8(r);
                dstRow[outOffset + 1] = Clip8(g);
                dstRow[outOffset + 2] = Clip8(b);
            }
        });
    }

    private static void ResampleVertical(RgbImage source, RgbImage destination, Filter filter)
    {
        int kernelSize = PrecomputeCoefficients(
            source.Height, destination.Height, filter, out int[] bounds, out int[] coefficients);

        int width = destination.Width;
        int outHeight = destination.Height;

        Parallelism.For(outHeight, y =>
        {
            int min = bounds[(y * 2) + 0];
            int count = bounds[(y * 2) + 1];
            int taps = y * kernelSize;
            Span<byte> dstRow = destination.Row(y);

            for (int x = 0; x < width; x++)
            {
                int offset = x * 3;
                int r = RoundingBias;
                int g = RoundingBias;
                int b = RoundingBias;

                for (int k = 0; k < count; k++)
                {
                    int weight = coefficients[taps + k];
                    ReadOnlySpan<byte> srcRow = source.Row(min + k);
                    r += srcRow[offset] * weight;
                    g += srcRow[offset + 1] * weight;
                    b += srcRow[offset + 2] * weight;
                }

                dstRow[offset] = Clip8(r);
                dstRow[offset + 1] = Clip8(g);
                dstRow[offset + 2] = Clip8(b);
            }
        });
    }

    private static double Evaluate(Filter filter, double x) =>
        filter == Filter.Lanczos ? LanczosFilter(x) : BicubicFilter(x);

    /// <summary>Pillow's three-lobe windowed-sinc kernel.</summary>
    private static double LanczosFilter(double x)
    {
        if (x is <= -LanczosLobes or >= LanczosLobes)
        {
            return 0.0;
        }

        return Sinc(x) * Sinc(x / LanczosLobes);
    }

    private static double Sinc(double x)
    {
        if (x == 0.0)
        {
            return 1.0;
        }

        double scaled = x * Math.PI;
        return Math.Sin(scaled) / scaled;
    }

    /// <summary>Pillow's bicubic kernel with <c>a = −0.5</c>.</summary>
    private static double BicubicFilter(double x)
    {
        if (x < 0.0)
        {
            x = -x;
        }

        if (x < 1.0)
        {
            return ((((BicubicA + 2.0) * x) - (BicubicA + 3.0)) * x * x) + 1.0;
        }

        if (x < 2.0)
        {
            return ((((x - 5.0) * x) + 8.0) * x - 4.0) * BicubicA;
        }

        return 0.0;
    }

    /// <summary>Pillow's saturating fixed-point to uint8 conversion.</summary>
    private static byte Clip8(int accumulator)
    {
        int value = accumulator >> PrecisionBits;
        if (value < 0)
        {
            return 0;
        }

        return value > 255 ? (byte)255 : (byte)value;
    }
}
