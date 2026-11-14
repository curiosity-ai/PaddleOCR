using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Models;

namespace PaddleOcrSharp.Tests.Core;

/// <summary>Tests for int8 weights: the quantiser, the kernels that read them, and how a checkpoint supplies them.</summary>
public class Int8WeightTests
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

    private static ushort[] ToBFloat16(float[] values) =>
        values.Select(value => (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16)).ToArray();

    private static float FromBFloat16(ushort bits) => BitConverter.UInt32BitsToSingle((uint)bits << 16);

    private static WeightMatrix BFloat16Matrix(float[] values, int rows, int cols) =>
        WeightMatrix.Create(MemoryMarshal.AsBytes<ushort>(ToBFloat16(values)).ToArray(), DType.BFloat16, rows, cols);

    /// <summary>Dequantised values from a matrix, through its own widening path.</summary>
    private static float[] Dequantised(WeightMatrix matrix)
    {
        float[] values = new float[matrix.Rows * matrix.Cols];
        matrix.CopyTo(values);
        return values;
    }

    [Theory]
    [InlineData(8, 256)]
    [InlineData(5, 1024)]
    [InlineData(3, 100)]
    public void QuantisingStaysWithinHalfAScalePerWeight(int rows, int cols)
    {
        float[] w = Gaussianish(rows * cols, seed: cols, scale: 0.03).Select(v => FromBFloat16(ToBFloat16([v])[0])).ToArray();
        WeightMatrix q = BFloat16Matrix(w, rows, cols).QuantizeInt8();

        Assert.Equal(DType.Int8, q.Dtype);
        int group = cols % 128 == 0 ? 128 : cols;
        Assert.Equal(group, q.GroupSize);

        float[] d = Dequantised(q);
        for (int r = 0; r < rows; r++)
        {
            for (int g = 0; g < cols / group; g++)
            {
                float peak = 0;
                for (int i = 0; i < group; i++)
                {
                    peak = Math.Max(peak, Math.Abs(w[(r * cols) + (g * group) + i]));
                }

                float scale = peak / 127f;
                for (int i = 0; i < group; i++)
                {
                    int k = (r * cols) + (g * group) + i;
                    Assert.True(Math.Abs(d[k] - w[k]) <= (scale * 0.5f) + (1e-6f * Math.Abs(w[k])), $"[{r},{g * group + i}]");
                }
            }
        }
    }

    /// <summary>Every GEMM path reads int8 weights as the matrix they dequantise to.</summary>
    [Theory]
    [InlineData(1, 1024, 130)]
    [InlineData(1, 256, 7)]
    [InlineData(5, 1024, 130)]
    [InlineData(3, 100, 9)]
    [InlineData(40, 1152, 96)]
    [InlineData(355, 1024, 50)]
    public void LinearMatchesTheDequantisedMatrix(int rows, int inner, int cols)
    {
        float[] x = Gaussianish(rows * inner, seed: rows, scale: 1);
        WeightMatrix q = BFloat16Matrix(Gaussianish(cols * inner, seed: cols, scale: 0.05), cols, inner).QuantizeInt8();
        float[] dequantised = Dequantised(q);
        float[] bias = Gaussianish(cols, 3, 0.1);

        float[] expected = new float[rows * cols];
        for (int m = 0; m < rows; m++)
        {
            for (int n = 0; n < cols; n++)
            {
                double sum = bias[n];
                for (int k = 0; k < inner; k++)
                {
                    sum += (double)x[(m * inner) + k] * dequantised[(n * inner) + k];
                }

                expected[(m * cols) + n] = (float)sum;
            }
        }

        float[] actual = new float[rows * cols];
        Gemm.Linear(x, rows, inner, q, bias, actual, cols);
        AssertClose(expected, actual, inner);

        if (rows >= GemmOuter.MinRows)
        {
            float[] outer = new float[rows * cols];
            GemmOuter.Linear(x, rows, inner, q, bias, outer, cols);
            AssertClose(expected, outer, inner);
        }
    }

    private static void AssertClose(float[] expected, float[] actual, int inner)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            double tolerance = 1e-5 + (Math.Abs(expected[i]) * 1e-5) + (Math.Sqrt(inner) * 2e-6);
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"[{i}] expected {expected[i]} got {actual[i]}");
        }
    }

    /// <summary>The projections of a layer fused into one call read int8 weights too.</summary>
    [Fact]
    public void FusedProjectionsAcceptInt8()
    {
        const int Rows = 6, Inner = 512;
        float[] x = Gaussianish(Rows * Inner, 1, 1);
        WeightMatrix[] weights = new[] { 200, 64, 64 }.Select((c, i) => BFloat16Matrix(Gaussianish(c * Inner, 10 + i, 0.05), c, Inner).QuantizeInt8()).ToArray();
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

    // ---- how a checkpoint supplies them ----

    private sealed record Entry(string Name, string Dtype, int[] Shape, byte[] Bytes);

    private static string WriteSafetensors(params Entry[] entries)
    {
        var header = new Dictionary<string, object>();
        long offset = 0;
        foreach (Entry e in entries)
        {
            header[e.Name] = new { dtype = e.Dtype, shape = e.Shape, data_offsets = new[] { offset, offset + e.Bytes.Length } };
            offset += e.Bytes.Length;
        }

        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        int pad = (8 - (json.Length % 8)) % 8;
        string path = Path.Combine(Path.GetTempPath(), $"int8-test-{Guid.NewGuid():N}.safetensors");
        using (var file = File.Create(path))
        {
            file.Write(BitConverter.GetBytes((long)(json.Length + pad)));
            file.Write(json);
            file.Write(Enumerable.Repeat((byte)' ', pad).ToArray());
            foreach (Entry e in entries)
            {
                file.Write(e.Bytes);
            }
        }

        return path;
    }

    /// <summary>
    /// A bfloat16 tensor is quantised on load only when the store asks for it and the caller says it
    /// may be; everything else is handed back as stored.
    /// </summary>
    [Fact]
    public void TheStoreQuantisesOnLoadOnlyWhenAskedAndAllowed()
    {
        float[] w = Gaussianish(16 * 256, 5, 0.03);
        string path = WriteSafetensors(new Entry("layer.weight", "BF16", [16, 256], MemoryMarshal.AsBytes<ushort>(ToBFloat16(w)).ToArray()));
        try
        {
            using (WeightStore plain = WeightStore.Open(path))
            {
                Assert.Equal(DType.BFloat16, plain.Matrix("layer.weight", WeightRole.Decoder).Dtype);
            }

            using (WeightStore asked = WeightStore.Open(path, WeightQuantization.Int8Compute))
            {
                Assert.Equal(DType.Int8, asked.Matrix("layer.weight", WeightRole.Decoder).Dtype);
                Assert.Equal(DType.BFloat16, asked.Matrix("layer.weight").Dtype);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// A checkpoint that stores int8 plus <c>name.scale</c> is read as stored, whatever the store was
    /// asked, and gives the same matrix the on-the-fly path would have made.
    /// </summary>
    [Theory]
    [InlineData(WeightQuantization.None)]
    [InlineData(WeightQuantization.Int8Compute)]
    public void AStoredInt8CheckpointIsReadAsStored(WeightQuantization asked)
    {
        const int Rows = 12, Cols = 384;
        float[] w = Gaussianish(Rows * Cols, 6, 0.04);
        WeightMatrix onTheFly = BFloat16Matrix(w, Rows, Cols).QuantizeInt8();

        // The same quantisation, written out the way the checkpoint stores it.
        float[] expected = Dequantised(onTheFly);
        int groups = Cols / onTheFly.GroupSize;
        var values = new byte[Rows * Cols];
        var scales = new float[Rows * groups];
        float[] source = ToBFloat16(w).Select(FromBFloat16).ToArray();
        for (int r = 0; r < Rows; r++)
        {
            for (int g = 0; g < groups; g++)
            {
                float peak = 0;
                for (int i = 0; i < onTheFly.GroupSize; i++)
                {
                    peak = Math.Max(peak, Math.Abs(source[(r * Cols) + (g * onTheFly.GroupSize) + i]));
                }

                float scale = peak / 127f;
                scales[(r * groups) + g] = scale;
                for (int i = 0; i < onTheFly.GroupSize; i++)
                {
                    int k = (r * Cols) + (g * onTheFly.GroupSize) + i;
                    values[k] = (byte)(sbyte)Math.Clamp(MathF.Round(source[k] / scale), -127f, 127f);
                }
            }
        }

        string path = WriteSafetensors(
            new Entry("layer.weight", "I8", [Rows, Cols], values),
            new Entry("layer.weight.scale", "F32", [Rows, groups], MemoryMarshal.AsBytes<float>(scales).ToArray()));
        try
        {
            using WeightStore store = WeightStore.Open(path, asked);
            WeightMatrix read = store.Matrix("layer.weight", WeightRole.Decoder);
            Assert.Equal(DType.Int8, read.Dtype);
            Assert.Equal(onTheFly.GroupSize, read.GroupSize);
            Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), Dequantised(read).Select(BitConverter.SingleToInt32Bits));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
