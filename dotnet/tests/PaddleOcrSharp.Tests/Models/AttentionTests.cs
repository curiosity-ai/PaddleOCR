using PaddleOcrSharp.Core;
using PaddleOcrSharp.Models.Vision;

namespace PaddleOcrSharp.Tests.Models;

/// <summary>
/// The vision tower's attention against a plain double-precision computation, at the sequence
/// lengths where its blocking has edges: shorter than a tile of keys, one over a tile, a block of
/// query rows with a remainder that is not a multiple of four, and a whole number of both.
/// </summary>
public class AttentionTests
{
    private const int HeadDim = 72;
    private const int RowStride = 80;

    private static float[] Gaussian(int count, int seed)
    {
        var random = new Random(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            double u = 1 - random.NextDouble();
            double v = random.NextDouble();
            values[i] = (float)(Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(2 * Math.PI * v));
        }

        return values;
    }

    /// <summary>
    /// 1, 3 and 37 fit inside one tile of keys; 128 is exactly one; 129 spills a single key into a
    /// second; 300 leaves a partial block of query rows after two full ones of 128; 1000 and 1030
    /// cross many tiles and end on remainders of 104 and 6 rows.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(37)]
    [InlineData(128)]
    [InlineData(129)]
    [InlineData(300)]
    [InlineData(1000)]
    [InlineData(1030)]
    public void MatchesADoublePrecisionReference(int tokens)
    {
        const int Heads = 3;
        float[] q = Gaussian(Heads * tokens * HeadDim, 1);
        float[] k = Gaussian(Heads * tokens * HeadDim, 2);
        float[] v = new float[Heads * tokens * RowStride];
        float[] raw = Gaussian(Heads * tokens * HeadDim, 3);
        for (int row = 0; row < Heads * tokens; row++)
        {
            Array.Copy(raw, row * HeadDim, v, row * RowStride, HeadDim);
        }

        float[] output = new float[v.Length];
        float scale = 1f / MathF.Sqrt(HeadDim);

        Attention.Bidirectional(q, k, v, output, Heads, tokens, HeadDim, scale, RowStride);

        double worst = 0;
        double[] weights = new double[tokens];
        for (int head = 0; head < Heads; head++)
        {
            int qk = head * tokens * HeadDim;
            int vo = head * tokens * RowStride;
            for (int row = 0; row < tokens; row++)
            {
                double max = double.NegativeInfinity;
                for (int t = 0; t < tokens; t++)
                {
                    double dot = 0;
                    for (int d = 0; d < HeadDim; d++)
                    {
                        dot += (double)q[qk + (row * HeadDim) + d] * k[qk + (t * HeadDim) + d];
                    }

                    weights[t] = dot * scale;
                    max = Math.Max(max, weights[t]);
                }

                double sum = 0;
                for (int t = 0; t < tokens; t++)
                {
                    weights[t] = Math.Exp(weights[t] - max);
                    sum += weights[t];
                }

                for (int d = 0; d < HeadDim; d++)
                {
                    double expected = 0;
                    for (int t = 0; t < tokens; t++)
                    {
                        expected += weights[t] / sum * v[vo + (t * RowStride) + d];
                    }

                    worst = Math.Max(worst, Math.Abs(expected - output[vo + (row * RowStride) + d]));
                }
            }
        }

        // float32 sums of up to a thousand terms of weights that add to one, through a
        // polynomial exp: a few parts in 1e6 of the largest output, with room to spare.
        Assert.True(worst < 5e-5, $"worst absolute error {worst:E2} over {tokens} tokens");
    }

    /// <summary>Which of the two layouts of the keys a call used must not show in its answer.</summary>
    [Fact]
    public void AnswerDoesNotDependOnHowTheWorkIsSplitAcrossThreads()
    {
        const int Tokens = 700;
        const int Heads = 4;
        float[] q = Gaussian(Heads * Tokens * HeadDim, 4);
        float[] k = Gaussian(Heads * Tokens * HeadDim, 5);
        float[] v = Gaussian(Heads * Tokens * RowStride, 6);
        float scale = 1f / MathF.Sqrt(HeadDim);

        float[] parallel = new float[v.Length];
        Attention.Bidirectional(q, k, v, parallel, Heads, Tokens, HeadDim, scale, RowStride);

        float[] serial = new float[v.Length];
        using (Parallelism.Use(new ParallelOptions { MaxDegreeOfParallelism = 1 }))
        {
            Attention.Bidirectional(q, k, v, serial, Heads, Tokens, HeadDim, scale, RowStride);
        }

        Assert.Equal(
            parallel.Select(BitConverter.SingleToInt32Bits),
            serial.Select(BitConverter.SingleToInt32Bits));
    }
}
