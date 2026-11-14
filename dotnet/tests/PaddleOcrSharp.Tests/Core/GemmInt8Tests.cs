using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>Tests for the integer-activation product, <see cref="GemmInt8"/>.</summary>
public class GemmInt8Tests
{
    private static float[] Gaussianish(int count, int seed, double scale)
    {
        var random = new Random(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            double u = 1 - random.NextDouble();
            double v = random.NextDouble();
            values[i] = (float)(Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(2 * Math.PI * v) * scale);
        }

        return values;
    }

    /// <summary>A per-row int8 matrix over random bfloat16-ish weights, marked as taking the integer path.</summary>
    private static WeightMatrix Weights(int rows, int cols, int seed)
    {
        ushort[] raw = Gaussianish(rows * cols, seed, 0.04)
            .Select(v => (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16)).ToArray();
        WeightMatrix bf16 = WeightMatrix.Create(MemoryMarshal.AsBytes<ushort>(raw).ToArray(), DType.BFloat16, rows, cols);
        return bf16.QuantizeInt8(groupSize: cols).WithComputeInt8();
    }

    /// <summary>What the integer product should give: quantise each activation row, then the exact sum.</summary>
    private static float[] Reference(float[] x, int rows, int inner, WeightMatrix weight, float[]? bias)
    {
        float[] w = new float[weight.Rows * inner];
        weight.CopyTo(w);
        float[] y = new float[rows * weight.Rows];
        for (int m = 0; m < rows; m++)
        {
            float peak = 0;
            for (int k = 0; k < inner; k++)
            {
                peak = Math.Max(peak, Math.Abs(x[(m * inner) + k]));
            }

            float scale = peak == 0 ? 1f : peak / 127f;
            for (int n = 0; n < weight.Rows; n++)
            {
                double sum = 0;
                for (int k = 0; k < inner; k++)
                {
                    double q = Math.Clamp(MathF.Round(x[(m * inner) + k] / scale), -127f, 127f);
                    sum += q * scale * w[(n * inner) + k];
                }

                y[(m * weight.Rows) + n] = (float)(sum + (bias?[n] ?? 0));
            }
        }

        return y;
    }

    [Theory]
    [InlineData(32, 1152, 8, false)]
    [InlineData(33, 1152, 12, true)]
    [InlineData(100, 1024, 13, true)]
    [InlineData(70, 4304, 21, false)]
    [InlineData(37, 4608, 130, true)]
    [InlineData(34, 80, 9, false)]
    [InlineData(34, 96, 9, true)]
    [InlineData(34, 112, 9, false)]
    [InlineData(40, 16, 130, true)]
    public void MatchesTheQuantisedReference(int rows, int inner, int cols, bool useBias)
    {
        if (!GemmInt8.Available)
        {
            return;
        }

        float[] x = Gaussianish(rows * inner, seed: rows, scale: 1);
        WeightMatrix weight = Weights(cols, inner, seed: cols);
        float[]? bias = useBias ? Gaussianish(cols, 9, 0.1) : null;

        float[] expected = Reference(x, rows, inner, weight, bias);
        float[] actual = new float[rows * cols];
        Gemm.Linear(x, rows, inner, weight, bias ?? [], actual, cols);

        for (int i = 0; i < expected.Length; i++)
        {
            double tolerance = 1e-5 + (Math.Abs(expected[i]) * 2e-6);
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"[{i}] expected {expected[i]} got {actual[i]}");
        }
    }

    /// <summary>
    /// The 256-bit and 512-bit tiles add the same integers in different orders, so their sums must be
    /// equal, and equal to a scalar sum, at every remainder a 64-byte step can leave: 0, 16, 32 and 48
    /// bytes, and the model's own reduction lengths (1152, 1024, 3072, 4304 with a remainder of 16,
    /// 4608).
    /// </summary>
    [Theory]
    [InlineData(16)]
    [InlineData(32)]
    [InlineData(48)]
    [InlineData(64)]
    [InlineData(80)]
    [InlineData(96)]
    [InlineData(112)]
    [InlineData(128)]
    [InlineData(1024)]
    [InlineData(1152)]
    [InlineData(3072)]
    [InlineData(4304)]
    [InlineData(4608)]
    public unsafe void WideAndNarrowTilesGiveTheSameIntegerSums(int k)
    {
        if (!GemmInt8.Available || !AvxVnni.V512.IsSupported)
        {
            return;
        }

        var random = new Random(k);
        byte[] x = new byte[4 * k];
        sbyte[] w = new sbyte[4 * k];
        random.NextBytes(x);
        for (int i = 0; i < w.Length; i++)
        {
            w[i] = (sbyte)random.Next(-127, 128);
        }

        // The extremes, where a sum is largest: every activation 255, every weight -127.
        byte[] xMax = Enumerable.Repeat((byte)255, 4 * k).ToArray();
        sbyte[] wMin = Enumerable.Repeat((sbyte)-127, 4 * k).ToArray();

        foreach ((byte[] xs, sbyte[] ws) in new[] { (x, w), (xMax, wMin) })
        {
            int[] expected = new int[16];
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int d = 0; d < k; d++)
                    {
                        expected[(i * 4) + j] += xs[(i * k) + d] * ws[(j * k) + d];
                    }
                }
            }

            int[] narrow = new int[16], wide = new int[16];
            fixed (byte* px = xs)
            fixed (sbyte* pw = ws)
            fixed (int* pn = narrow, pwide = wide)
            {
                GemmInt8.Tile4x4(px, k, pw, k, pn);
                GemmInt8.Tile4x4Wide(px, k, pw, k, pwide);
            }

            Assert.Equal(expected, narrow);
            Assert.Equal(expected, wide);

            // Four rows against one weight row, the ragged-panel tile.
            for (int j = 0; j < 4; j++)
            {
                int[] oneNarrow = new int[4], oneWide = new int[4];
                fixed (byte* px = xs)
                fixed (sbyte* pw = ws)
                fixed (int* pn = oneNarrow, pwide = oneWide)
                {
                    GemmInt8.Tile4x1(px, k, pw + (j * k), pn);
                    GemmInt8.Tile4x1Wide(px, k, pw + (j * k), pwide);
                }

                int[] column = Enumerable.Range(0, 4).Select(i => expected[(i * 4) + j]).ToArray();
                Assert.Equal(column, oneNarrow);
                Assert.Equal(column, oneWide);
            }
        }
    }

    /// <summary>Fused projections quantise the activations once and give what separate calls give, bit for bit.</summary>
    [Fact]
    public void FusedProjectionsMatchSeparateCalls()
    {
        if (!GemmInt8.Available)
        {
            return;
        }

        const int Rows = 45, Inner = 1152;
        float[] x = Gaussianish(Rows * Inner, 1, 1);
        WeightMatrix[] weights = new[] { 200, 64, 31 }.Select((c, i) => Weights(c, Inner, 10 + i)).ToArray();
        float[][] separate = weights.Select(w => new float[Rows * w.Rows]).ToArray();
        float[][] fused = weights.Select(w => new float[Rows * w.Rows]).ToArray();

        for (int i = 0; i < weights.Length; i++)
        {
            Gemm.Linear(x, Rows, Inner, weights[i], default, separate[i], weights[i].Rows);
        }

        Gemm.LinearMany(x, Rows, Inner, weights.Select((w, i) => new Gemm.LinearPart(w, default, fused[i])).ToArray());

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.Equal(separate[i].Select(BitConverter.SingleToInt32Bits), fused[i].Select(BitConverter.SingleToInt32Bits));
        }
    }

    /// <summary>The path applies only to per-row int8 weights that opted in, with enough rows.</summary>
    [Fact]
    public void AppliesOnlyWhereAskedFor()
    {
        WeightMatrix perRow = Weights(16, 256, 1);
        Assert.Equal(GemmInt8.Available, GemmInt8.Applies(GemmInt8.MinRows, perRow));
        Assert.False(GemmInt8.Applies(GemmInt8.MinRows - 1, perRow));

        WeightMatrix notMarked = WeightMatrix.CreateInt8(new byte[16 * 256], new float[16], 256, 16, 256);
        Assert.False(GemmInt8.Applies(GemmInt8.MinRows, notMarked));

        WeightMatrix grouped = WeightMatrix.CreateInt8(new byte[16 * 256], new float[32], 128, 16, 256).WithComputeInt8();
        Assert.False(GemmInt8.Applies(GemmInt8.MinRows, grouped));
    }

    /// <summary>The row sums are the sums of each row's signed values.</summary>
    [Fact]
    public void RowSumsAreTheSumsOfTheSignedValues()
    {
        WeightMatrix weight = Weights(9, 100, 4);
        int[] expected = new int[9];
        ReadOnlySpan<byte> values = weight.Int8Values;
        for (int r = 0; r < 9; r++)
        {
            for (int c = 0; c < 100; c++)
            {
                expected[r] += (sbyte)values[(r * 100) + c];
            }
        }

        Assert.Equal(expected, weight.GetRowSums().ToArray());
    }
}
