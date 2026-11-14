using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats.Paddle;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Matrix products, normalisations and softmax.</summary>
internal static class LinearOps
{
    /// <summary>
    /// Batched matrix product with optional transposes, matching <c>pd_op.matmul</c>.
    /// </summary>
    public static PaddleTensor MatMul(PaddleTensor x, PaddleTensor y, bool transposeX, bool transposeY)
    {
        int[] xShape = Promote(x.Shape);
        int[] yShape = Promote(y.Shape);

        int xRows = transposeX ? xShape[^1] : xShape[^2];
        int xCols = transposeX ? xShape[^2] : xShape[^1];
        int yRows = transposeY ? yShape[^1] : yShape[^2];
        int yCols = transposeY ? yShape[^2] : yShape[^1];

        if (xCols != yRows)
        {
            throw new InvalidOperationException(
                $"matmul shape mismatch: [{string.Join(",", x.Shape)}] x [{string.Join(",", y.Shape)}] " +
                $"(transposeX={transposeX}, transposeY={transposeY}).");
        }

        int[] xBatch = xShape[..^2];
        int[] yBatch = yShape[..^2];
        int[] batchShape = Broadcast.ResultShape(xBatch, yBatch);
        int batches = PaddleTensor.ElementCount(batchShape);

        int[] resultShape = [.. batchShape, xRows, yCols];
        PaddleTensor result = PaddleTensor.Float(resultShape);

        int[] xBatchStrides = Broadcast.StridesFor(xBatch, batchShape);
        int[] yBatchStrides = Broadcast.StridesFor(yBatch, batchShape);

        int xMatrix = xShape[^2] * xShape[^1];
        int yMatrix = yShape[^2] * yShape[^1];
        int outMatrix = xRows * yCols;

        ReadOnlyMemory<float> xData = x.FloatMemory;
        ReadOnlyMemory<float> yData = y.FloatMemory;
        Memory<float> output = result.FloatMemory;

        int[] counters = new int[batchShape.Length];
        for (int b = 0; b < batches; b++)
        {
            int xOffset = 0;
            int yOffset = 0;
            for (int axis = 0; axis < batchShape.Length; axis++)
            {
                xOffset += counters[axis] * xBatchStrides[axis];
                yOffset += counters[axis] * yBatchStrides[axis];
            }

            Gemm.MatMul(
                xData.Slice(xOffset * xMatrix, xMatrix),
                xRows,
                xCols,
                transposeX,
                yData.Slice(yOffset * yMatrix, yMatrix),
                yCols,
                transposeY,
                output.Slice(b * outMatrix, outMatrix));

            for (int axis = batchShape.Length - 1; axis >= 0; axis--)
            {
                if (++counters[axis] < batchShape[axis])
                {
                    break;
                }

                counters[axis] = 0;
            }
        }

        // Paddle drops the padded dimension when an input was rank-1.
        if (x.Rank == 1 && y.Rank == 1)
        {
            return result.Reshaped([]);
        }

        if (x.Rank == 1)
        {
            var shape = new List<int>(resultShape);
            shape.RemoveAt(shape.Count - 2);
            return result.Reshaped([.. shape]);
        }

        if (y.Rank == 1)
        {
            var shape = new List<int>(resultShape);
            shape.RemoveAt(shape.Count - 1);
            return result.Reshaped([.. shape]);
        }

        return result;
    }

    /// <summary>Batched matrix product without transposes, matching <c>pd_op.bmm</c>.</summary>
    public static PaddleTensor Bmm(PaddleTensor x, PaddleTensor y) => MatMul(x, y, false, false);

    /// <summary>
    /// Layer normalisation over the trailing dimensions from <paramref name="beginNormAxis"/>.
    /// </summary>
    public static PaddleTensor LayerNorm(
        PaddleTensor input,
        PaddleTensor? scale,
        PaddleTensor? bias,
        float epsilon,
        int beginNormAxis)
    {
        int width = 1;
        for (int i = beginNormAxis; i < input.Rank; i++)
        {
            width *= input.Shape[i];
        }

        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        input.FloatSpan.CopyTo(result.FloatSpan);

        Norms.LayerNorm(
            result.FloatSpan,
            width,
            scale is not null ? scale.FloatSpan : OnesFor(width),
            bias is not null ? bias.FloatSpan : ReadOnlySpan<float>.Empty,
            epsilon);

        return result;
    }

    private static float[] OnesFor(int width)
    {
        float[] ones = new float[width];
        Array.Fill(ones, 1f);
        return ones;
    }

    /// <summary>Softmax along <paramref name="axis"/>.</summary>
    public static PaddleTensor Softmax(PaddleTensor input, int axis)
    {
        int rank = input.Rank;
        axis = axis < 0 ? axis + rank : axis;

        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);
        input.FloatSpan.CopyTo(result.FloatSpan);

        int width = input.Shape[axis];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++)
        {
            inner *= input.Shape[i];
        }

        int outer = width == 0 ? 0 : input.Count / (width * inner);
        Span<float> data = result.FloatSpan;

        if (inner == 1)
        {
            for (int o = 0; o < outer; o++)
            {
                Kernels.Softmax(data.Slice(o * width, width));
            }

            return result;
        }

        Span<float> scratch = new float[width];
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                int baseIndex = (o * width * inner) + i;
                for (int w = 0; w < width; w++)
                {
                    scratch[w] = data[baseIndex + (w * inner)];
                }

                Kernels.Softmax(scratch);

                for (int w = 0; w < width; w++)
                {
                    data[baseIndex + (w * inner)] = scratch[w];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Inference-time batch normalisation:
    /// <c>(x − mean) / sqrt(var + eps) · scale + bias</c> over the channel dimension.
    /// </summary>
    public static PaddleTensor BatchNorm(
        PaddleTensor input,
        PaddleTensor mean,
        PaddleTensor variance,
        PaddleTensor scale,
        PaddleTensor bias,
        float epsilon,
        string dataFormat)
    {
        if (dataFormat != "NCHW")
        {
            throw new NotSupportedException($"batch_norm data format '{dataFormat}' is not supported.");
        }

        int channels = input.Shape[1];
        int spatial = 1;
        for (int i = 2; i < input.Rank; i++)
        {
            spatial *= input.Shape[i];
        }

        int batch = input.Shape[0];
        PaddleTensor result = PaddleTensor.Float([.. input.Shape]);

        ReadOnlySpan<float> source = input.FloatSpan;
        Span<float> destination = result.FloatSpan;
        ReadOnlySpan<float> means = mean.FloatSpan;
        ReadOnlySpan<float> variances = variance.FloatSpan;
        ReadOnlySpan<float> scales = scale.FloatSpan;
        ReadOnlySpan<float> biases = bias.FloatSpan;

        // Per-channel affine constants, pooled: two arrays a call is 12 KiB, and this operator
        // runs 105 times over a layout detection.
        using PooledBuffer constants = TensorPool.Rent(channels * 2);
        Span<float> multiplier = constants.Span[..channels];
        Span<float> offset = constants.Span.Slice(channels, channels);
        for (int c = 0; c < channels; c++)
        {
            float inverse = 1f / MathF.Sqrt(variances[c] + epsilon);
            multiplier[c] = scales[c] * inverse;
            offset[c] = biases[c] - (means[c] * multiplier[c]);
        }

        // At inference this is one affine pass per channel plane, and the planes are independent,
        // so it threads over them and each is a vector multiply-add. Element at a time and on one
        // thread — which is what it was — the operator cost 156 ms of a 3.5 s detection.
        Scale(input.Floats!, result.Floats!, multiplier, offset, batch, channels, spatial);

        return result;
    }

    /// <summary>Applies one multiplier and offset per channel plane, planes in parallel.</summary>
    private static void Scale(
        float[] source,
        float[] destination,
        ReadOnlySpan<float> multiplier,
        ReadOnlySpan<float> offset,
        int batch,
        int channels,
        int spatial)
    {
        // Copied out because a lambda may not close over a span.
        float[] scale = TensorPool.RentArray(channels);
        float[] shift = TensorPool.RentArray(channels);
        multiplier.CopyTo(scale);
        offset.CopyTo(shift);

        Parallelism.For(batch * channels, plane =>
        {
            int c = plane % channels;
            int start = plane * spatial;
            Affine(source.AsSpan(start, spatial), scale[c], shift[c], destination.AsSpan(start, spatial));
        });

        TensorPool.Return(scale);
        TensorPool.Return(shift);
    }

    /// <summary><c>y = x * scale + shift</c> over one plane.</summary>
    /// <remarks>
    /// `TensorPrimitives` has no scalar-multiplier multiply-add, and doing it as a
    /// multiply then an add is two passes over the plane where the arithmetic needs one.
    /// </remarks>
    private static void Affine(ReadOnlySpan<float> source, float scale, float shift, Span<float> destination)
    {
        int i = 0;

        if (Simd.Use256 && source.Length >= Vector256<float>.Count)
        {
            Vector256<float> m = Vector256.Create(scale);
            Vector256<float> o = Vector256.Create(shift);
            ref float x = ref MemoryMarshal.GetReference(source);
            ref float y = ref MemoryMarshal.GetReference(destination);

            for (; i <= source.Length - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                Vector256.FusedMultiplyAdd(Vector256.LoadUnsafe(ref x, (nuint)i), m, o)
                    .StoreUnsafe(ref y, (nuint)i);
            }
        }

        for (; i < source.Length; i++)
        {
            destination[i] = (source[i] * scale) + shift;
        }
    }

    private static int[] Promote(int[] shape) => shape.Length switch
    {
        0 => [1, 1],
        1 => [1, shape[0]],
        _ => shape,
    };
}
