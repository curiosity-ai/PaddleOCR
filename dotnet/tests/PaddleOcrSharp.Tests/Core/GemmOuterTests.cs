using System.Runtime.InteropServices;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>
/// Checks the outer-product form of <see cref="Gemm.Linear"/> against a float64 reference.
/// </summary>
/// <remarks>
/// It is not bit-identical to the dot form, so what is pinned is accuracy: each output must be
/// within a few float32 ulps of the exact sum, scaled by the magnitude of the terms (a sum that
/// cancels to nearly zero has an error set by what it cancelled, not by its result).
/// </remarks>
public class GemmOuterTests
{
    private static float[] Random(int count, int seed, double scale = 1)
    {
        var random = new Random(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = (float)(((random.NextDouble() * 2) - 1) * scale);
        }

        return values;
    }

    private static ushort ToBFloat16(float value) =>
        (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);

    private static float FromBFloat16(ushort bits) =>
        BitConverter.UInt32BitsToSingle((uint)bits << 16);

    [Theory]
    [InlineData(32, 1152, 8, false)]
    [InlineData(33, 1152, 12, false)]
    [InlineData(100, 27, 13, false)]
    [InlineData(355, 1024, 130, true)]
    [InlineData(64, 4304, 36, true)]
    [InlineData(70, 4608, 21, false)]
    [InlineData(1368, 1152, 116, true)]
    public void MatchesAFloat64Reference(int rows, int inner, int cols, bool bfloat16)
    {
        float[] x = Random(rows * inner, seed: rows);
        float[] bias = Random(cols, seed: 7);
        float[] w = Random(cols * inner, seed: cols, scale: 0.05);

        WeightMatrix weight;
        if (bfloat16)
        {
            ushort[] raw = w.Select(ToBFloat16).ToArray();
            w = raw.Select(FromBFloat16).ToArray();
            weight = WeightMatrix.Create(MemoryMarshal.AsBytes<ushort>(raw).ToArray(), DType.BFloat16, cols, inner);
        }
        else
        {
            weight = WeightMatrix.FromFloats(w, cols, inner);
        }

        float[] y = new float[rows * cols];
        GemmOuter.Linear(x, rows, inner, weight, bias, y, cols);

        for (int m = 0; m < rows; m++)
        {
            for (int n = 0; n < cols; n++)
            {
                double exact = bias[n];
                double magnitude = Math.Abs((double)bias[n]);
                for (int k = 0; k < inner; k++)
                {
                    double term = (double)x[(m * inner) + k] * w[(n * inner) + k];
                    exact += term;
                    magnitude += Math.Abs(term);
                }

                // A sequential float32 chain's error grows with the length of the chain.
                double tolerance = magnitude * 1.2e-7 * Math.Max(8, Math.Sqrt(inner)) + 1e-6;
                Assert.True(
                    Math.Abs(y[(m * cols) + n] - exact) <= tolerance,
                    $"[{m},{n}] got {y[(m * cols) + n]} expected {exact} (tolerance {tolerance})");
            }
        }
    }

    /// <summary>The same input gives the same bits twice: the order is fixed, whatever the threading.</summary>
    [Fact]
    public void IsDeterministic()
    {
        const int Rows = 129, Inner = 2500, Cols = 50;
        float[] x = Random(Rows * Inner, 1);
        WeightMatrix weight = WeightMatrix.FromFloats(Random(Cols * Inner, 2, 0.05), Cols, Inner);

        float[] first = new float[Rows * Cols];
        float[] second = new float[Rows * Cols];
        GemmOuter.Linear(x, Rows, Inner, weight, default, first, Cols);
        GemmOuter.Linear(x, Rows, Inner, weight, default, second, Cols);

        Assert.Equal(first.Select(BitConverter.SingleToInt32Bits), second.Select(BitConverter.SingleToInt32Bits));
    }

    /// <summary>
    /// Few rows are not worth the padding and stay on the dot form, and the whole path is off unless
    /// <c>PADDLEOCR_SHARP_GEMM=outer</c> is set, so the default build's output is unchanged.
    /// </summary>
    [Fact]
    public void FewRowsDoNotTakeThePath()
    {
        Assert.False(GemmOuter.Applies(1));
        Assert.False(GemmOuter.Applies(GemmOuter.MinRows - 1));

        bool requested = string.Equals(
            Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_GEMM"), "outer", StringComparison.OrdinalIgnoreCase);
        Assert.Equal(requested, GemmOuter.Enabled);
    }
}
