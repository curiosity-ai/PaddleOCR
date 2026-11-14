using System.Runtime.InteropServices;
using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>
/// <see cref="Gemm.LinearMany"/> computes several projections of one input; each must come out
/// exactly as it would from a call of its own, in either accumulation order.
/// </summary>
public class GemmManyTests
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

    private static WeightMatrix Weight(int cols, int inner, int seed, bool bfloat16)
    {
        float[] w = Random(cols * inner, seed, 0.05);
        if (!bfloat16)
        {
            return WeightMatrix.FromFloats(w, cols, inner);
        }

        ushort[] raw = w.Select(value => (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16)).ToArray();
        return WeightMatrix.Create(MemoryMarshal.AsBytes<ushort>(raw).ToArray(), DType.BFloat16, cols, inner);
    }

    private static IEnumerable<int> Bits(float[] values) => values.Select(BitConverter.SingleToInt32Bits);

    // The query / key / value shape of the decoder: one wide projection and two narrow ones.
    private static readonly int[] Widths = [2048, 256, 130];

    [Theory]
    [InlineData(1, 1024, true)]
    [InlineData(5, 1024, true)]
    [InlineData(31, 1152, false)]
    [InlineData(40, 1152, true)]
    [InlineData(355, 1024, true)]
    public void FusedProjectionsMatchSeparateCalls(int rows, int inner, bool bfloat16)
    {
        float[] x = Random(rows * inner, seed: rows);
        WeightMatrix[] weights = Widths.Select((cols, i) => Weight(cols, inner, 10 + i, bfloat16)).ToArray();
        float[][] biases = [Random(Widths[0], 90), [], Random(Widths[2], 92)];

        float[][] expected = Widths.Select(cols => new float[rows * cols]).ToArray();
        float[][] fused = Widths.Select(cols => new float[rows * cols]).ToArray();

        for (int i = 0; i < Widths.Length; i++)
        {
            Gemm.Linear(x, rows, inner, weights[i], biases[i], expected[i], Widths[i]);
        }

        Gemm.LinearMany(
            x,
            rows,
            inner,
            weights.Select((w, i) => new Gemm.LinearPart(w, biases[i], fused[i])).ToArray());

        for (int i = 0; i < Widths.Length; i++)
        {
            Assert.Equal(Bits(expected[i]), Bits(fused[i]));
        }
    }

    /// <summary>The same through the outer-product path, called directly so that it is exercised whatever the environment says.</summary>
    [Theory]
    [InlineData(32, 1152, false)]
    [InlineData(70, 4304, true)]
    [InlineData(355, 1024, true)]
    public void FusedProjectionsMatchSeparateCallsInTheOuterForm(int rows, int inner, bool bfloat16)
    {
        float[] x = Random(rows * inner, seed: rows);
        WeightMatrix[] weights = Widths.Select((cols, i) => Weight(cols, inner, 20 + i, bfloat16)).ToArray();
        float[][] biases = [Random(Widths[0], 91), Random(Widths[1], 92), []];

        float[][] expected = Widths.Select(cols => new float[rows * cols]).ToArray();
        float[][] fused = Widths.Select(cols => new float[rows * cols]).ToArray();

        for (int i = 0; i < Widths.Length; i++)
        {
            GemmOuter.Linear(x, rows, inner, weights[i], biases[i], expected[i], Widths[i]);
        }

        GemmOuter.LinearMany(
            x,
            rows,
            inner,
            weights.Select((w, i) => new Gemm.LinearPart(w, biases[i], fused[i])).ToArray());

        for (int i = 0; i < Widths.Length; i++)
        {
            Assert.Equal(Bits(expected[i]), Bits(fused[i]));
        }
    }

    /// <summary>A mismatched part is rejected like a mismatched single call.</summary>
    [Fact]
    public void AMismatchedPartIsRejected()
    {
        float[] x = Random(4 * 64, 1);
        WeightMatrix wrong = Weight(8, 32, 2, bfloat16: false);
        Assert.Throws<ArgumentException>(() =>
            Gemm.LinearMany(x, 4, 64, [new Gemm.LinearPart(wrong, default, new float[4 * 8])]));
    }
}
