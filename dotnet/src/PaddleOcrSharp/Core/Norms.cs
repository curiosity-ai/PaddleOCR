using System.Numerics.Tensors;

namespace PaddleOcrSharp.Core;

/// <summary>Normalisation layers, computed in float32 to match upstream precision.</summary>
public static class Norms
{
    /// <summary>
    /// Root-mean-square normalisation: <c>x · rsqrt(mean(x²) + eps) · weight</c>.
    /// </summary>
    /// <remarks>
    /// Port of <c>Ernie4_5RMSNorm.forward</c>. Upstream casts to float32 for the variance and
    /// keeps the scale multiply outside that cast, which is exactly what this does.
    /// </remarks>
    /// <param name="values">Rows of <paramref name="width"/> elements, normalised in place.</param>
    /// <param name="width">Size of the normalised dimension.</param>
    /// <param name="weight">Per-channel scale of length <paramref name="width"/>.</param>
    /// <param name="epsilon">Variance epsilon.</param>
    public static void RmsNorm(Span<float> values, int width, ReadOnlySpan<float> weight, float epsilon)
    {
        int rows = values.Length / width;
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = values.Slice(r * width, width);

            float sumSquares = 0f;
            for (int i = 0; i < width; i++)
            {
                float v = row[i];
                sumSquares += v * v;
            }

            float scale = 1f / MathF.Sqrt((sumSquares / width) + epsilon);
            for (int i = 0; i < width; i++)
            {
                row[i] = row[i] * scale * weight[i];
            }
        }
    }

    /// <summary>
    /// Layer normalisation: <c>(x − mean) / sqrt(var + eps) · weight + bias</c>, with the
    /// biased (population) variance PyTorch uses.
    /// </summary>
    public static void LayerNorm(
        Span<float> values,
        int width,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        float epsilon)
    {
        int rows = values.Length / width;
        for (int r = 0; r < rows; r++)
        {
            Span<float> row = values.Slice(r * width, width);

            float sum = 0f;
            for (int i = 0; i < width; i++)
            {
                sum += row[i];
            }

            float mean = sum / width;

            float sumSquares = 0f;
            for (int i = 0; i < width; i++)
            {
                float d = row[i] - mean;
                sumSquares += d * d;
            }

            float scale = 1f / MathF.Sqrt((sumSquares / width) + epsilon);

            if (bias.IsEmpty)
            {
                for (int i = 0; i < width; i++)
                {
                    row[i] = (row[i] - mean) * scale * weight[i];
                }
            }
            else
            {
                for (int i = 0; i < width; i++)
                {
                    row[i] = ((row[i] - mean) * scale * weight[i]) + bias[i];
                }
            }
        }
    }

    /// <summary>
    /// Layer normalisation over many rows, parallelised. Rows are independent, so this is a
    /// straight split of <see cref="LayerNorm"/>.
    /// </summary>
    public static void LayerNormParallel(
        Memory<float> values,
        int width,
        ReadOnlyMemory<float> weight,
        ReadOnlyMemory<float> bias,
        float epsilon)
    {
        int rows = values.Length / width;
        if (rows < 64)
        {
            LayerNorm(values.Span, width, weight.Span, bias.Span, epsilon);
            return;
        }

        int threads = Environment.ProcessorCount;
        int chunk = (rows + threads - 1) / threads;
        Parallelism.For((rows + chunk - 1) / chunk, block =>
        {
            int start = block * chunk;
            int count = Math.Min(chunk, rows - start);
            LayerNorm(
                values.Span.Slice(start * width, count * width),
                width,
                weight.Span,
                bias.Span,
                epsilon);
        });
    }

    /// <summary>
    /// Root-mean-square normalisation over many rows, parallelised.
    /// </summary>
    public static void RmsNormParallel(
        Memory<float> values,
        int width,
        ReadOnlyMemory<float> weight,
        float epsilon)
    {
        int rows = values.Length / width;
        if (rows < 64)
        {
            RmsNorm(values.Span, width, weight.Span, epsilon);
            return;
        }

        int threads = Environment.ProcessorCount;
        int chunk = (rows + threads - 1) / threads;
        Parallelism.For((rows + chunk - 1) / chunk, block =>
        {
            int start = block * chunk;
            int count = Math.Min(chunk, rows - start);
            RmsNorm(values.Span.Slice(start * width, count * width), width, weight.Span, epsilon);
        });
    }

    /// <summary>Sum of squares of <paramref name="values"/>, for diagnostics and tests.</summary>
    public static float SumOfSquares(ReadOnlySpan<float> values) =>
        TensorPrimitives.SumOfSquares(values);
}
