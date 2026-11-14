using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Formats;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Convolution and pooling, in NCHW layout.</summary>
internal static class ConvOps
{

    /// <summary>
    /// 2-D convolution, matching <c>pd_op.conv2d</c> and <c>pd_op.depthwise_conv2d</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// im2col followed by <see cref="Gemm.Linear"/>, over a block of output rows at a time.
    /// Depthwise convolutions go direct instead: with one channel per group, im2col would
    /// materialise a column buffer far larger than the work it saves.
    /// </para>
    /// <para>
    /// The GEMM is the whole cost of this operator — 157 GFLOP of it in one layout detection —
    /// so which kernel runs it decides the stage. Written the obvious way, one product per
    /// output row through <see cref="Gemm.MatMul"/>, it measured 81 GFLOP/s against the 250 the
    /// same machine sustains through <see cref="Gemm.Linear"/> on the vision tower's shapes. The
    /// difference is the inner loop: <c>MatMul</c>'s <c>k × n</c> form accumulates whole output
    /// rows in memory, one load and one store per multiply-add, where <c>Linear</c> widens a
    /// column panel once and reduces four rows against four columns in registers. So the
    /// convolution is posed as the shape <c>Linear</c> wants — filters as the activation rows,
    /// the im2col columns as the weight panel — which also lands the product in
    /// <c>[channel, pixel]</c> order, the layout the NCHW result already has.
    /// </para>
    /// <para>
    /// A 1x1 convolution looks like it should skip im2col altogether — its columns are a
    /// transpose of the input and nothing else, 382 MB of gathering a detection — and hand the
    /// operands to <see cref="Gemm.MatMul"/>'s <c>k x n</c> form where they lie. Measured, that
    /// is slower: conv2d 1478 -> 1686 ms. The transpose buys the reduction a contiguous axis,
    /// and <c>Linear</c>'s four-by-four tile does sixteen multiply-adds per eight loads where
    /// the broadcast kernel does eight per six.
    /// </para>
    /// <para>
    /// A block is sized to keep its columns near <see cref="ColumnBudgetBytes"/>: the panel loop
    /// re-reads the filters once per panel, so a block wants to be small enough that they stay
    /// cached, and large enough that the per-block panel widening is amortised.
    /// </para>
    /// </remarks>
    public static PaddleTensor Conv2d(
        PaddleTensor input,
        PaddleTensor weight,
        int[] strides,
        int[] paddings,
        int[] dilations,
        int groups,
        string paddingAlgorithm,
        string dataFormat)
    {
        if (dataFormat != "NCHW")
        {
            throw new NotSupportedException($"conv2d data format '{dataFormat}' is not supported.");
        }

        int batch = input.Shape[0];
        int inChannels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outChannels = weight.Shape[0];
        int kernelHeight = weight.Shape[2];
        int kernelWidth = weight.Shape[3];

        (int padTop, int padBottom, int padLeft, int padRight) = ResolvePadding(
            paddingAlgorithm,
            paddings,
            inHeight,
            inWidth,
            kernelHeight,
            kernelWidth,
            strides,
            dilations);

        int outHeight = ((inHeight + padTop + padBottom - (dilations[0] * (kernelHeight - 1)) - 1) / strides[0]) + 1;
        int outWidth = ((inWidth + padLeft + padRight - (dilations[1] * (kernelWidth - 1)) - 1) / strides[1]) + 1;

        PaddleTensor result = PaddleTensor.Float([batch, outChannels, outHeight, outWidth]);
        if (result.Count == 0)
        {
            return result;
        }

        if (groups == inChannels && groups == outChannels)
        {
            Depthwise(
                input, weight, result,
                batch, inChannels, inHeight, inWidth,
                kernelHeight, kernelWidth, outHeight, outWidth,
                strides, dilations, padTop, padLeft);
            return result;
        }

        int inGroupChannels = inChannels / groups;
        int outGroupChannels = outChannels / groups;
        int patch = inGroupChannels * kernelHeight * kernelWidth;
        int plane = outHeight * outWidth;

        float[] weightArray = weight.Floats!;

        // Rows per block, from the column budget: one row of columns is outWidth * patch floats.
        int rowBytes = outWidth * patch * sizeof(float);
        int blockRows = Math.Clamp(ColumnBudgetBytes / Math.Max(1, rowBytes), 1, outHeight);

        int columnCount = blockRows * outWidth * patch;
        int tileCount = outGroupChannels * blockRows * outWidth;

        // Bytes rather than floats: the columns are written as floats and read by the GEMM as a
        // weight matrix, which is a view over bytes, and renting them this way keeps that view
        // free of a copy.
        byte[] columnBytes = TensorPool.RentBytes(columnCount * sizeof(float));
        columnBytes.AsSpan(0, columnCount * sizeof(float)).Clear();
        using PooledBuffer tile = TensorPool.Rent(tileCount);

        for (int n = 0; n < batch; n++)
        {
            for (int g = 0; g < groups; g++)
            {
                int inputBase = ((n * inChannels) + (g * inGroupChannels)) * inHeight * inWidth;
                int outputBase = ((n * outChannels) + (g * outGroupChannels)) * plane;
                int weightBase = g * outGroupChannels * patch;

                for (int rowStart = 0; rowStart < outHeight; rowStart += blockRows)
                {
                    int rows = Math.Min(blockRows, outHeight - rowStart);
                    int pixels = rows * outWidth;

                    Im2ColBlock(
                        input.Floats!, inputBase, inGroupChannels, inHeight, inWidth,
                        kernelHeight, kernelWidth, strides, dilations, padTop, padLeft,
                        rowStart, rows, outWidth, patch, columnBytes);

                    // y[outGroupChannels, pixels] = filters[outGroupChannels, patch]
                    //                             · columnsᵀ[patch, pixels]
                    Gemm.Linear(
                        weightArray.AsMemory(weightBase, outGroupChannels * patch),
                        outGroupChannels,
                        patch,
                        WeightMatrix.Create(
                            columnBytes.AsMemory(0, pixels * patch * sizeof(float)),
                            DType.Float32,
                            pixels,
                            patch),
                        ReadOnlyMemory<float>.Empty,
                        tile.Memory[..(outGroupChannels * pixels)],
                        pixels);

                    // Each output channel's block is one contiguous run of the result plane.
                    for (int oc = 0; oc < outGroupChannels; oc++)
                    {
                        tile.Span.Slice(oc * pixels, pixels).CopyTo(
                            result.Floats.AsSpan(outputBase + (oc * plane) + (rowStart * outWidth), pixels));
                    }
                }
            }
        }

        TensorPool.ReturnBytes(columnBytes);
        return result;
    }

    /// <summary>Columns budget for one block of output rows, before the GEMM runs over it.</summary>
    private const int ColumnBudgetBytes = 16 << 20;

    /// <summary>
    /// Fills a block of output rows' im2col columns, laid out <c>pixel x patch</c> — the row-major
    /// <c>[n, k]</c> form <see cref="Gemm.Linear"/> reads its weight panel in.
    /// </summary>
    /// <remarks>
    /// The buffer is zeroed once, when it is rented, and the positions a tap skips because it
    /// falls outside the image are left alone rather than rewritten: whether a tap is in range
    /// depends on the output pixel and the tap, both of which this fill visits exactly once per
    /// block, so a skipped position is simply one this fill never writes. What the previous block
    /// left there is the problem instead, so a skipped position is cleared as it is passed.
    /// </remarks>
    private static void Im2ColBlock(
        float[] input,
        int inputBase,
        int channels,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int[] strides,
        int[] dilations,
        int padTop,
        int padLeft,
        int rowStart,
        int rows,
        int outWidth,
        int patch,
        byte[] columns)
    {
        int strideY = strides[0];
        int strideX = strides[1];
        int dilationY = dilations[0];
        int dilationX = dilations[1];

        // Threaded over the block's rows: this fill is 2.1 GB of stores over a detection, and the
        // GEMM that follows spreads itself, so leaving the fill serial hands the stage back
        // whatever the GEMM saves.
        FillRows(
            input, inputBase, channels, inHeight, inWidth,
            kernelHeight, kernelWidth, strideY, strideX, dilationY, dilationX, padTop, padLeft,
            rowStart, rows, outWidth, patch, columns);
    }

    /// <summary>The body of <see cref="Im2ColBlock"/>, one output row at a time.</summary>
    /// <remarks>
    /// Separate so the lambda's display class is allocated only where it is used —
    /// <see cref="Conv2d"/> runs ninety times over a layout graph.
    /// </remarks>
    private static void FillRows(
        float[] input,
        int inputBase,
        int channels,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int strideY,
        int strideX,
        int dilationY,
        int dilationX,
        int padTop,
        int padLeft,
        int rowStart,
        int rows,
        int outWidth,
        int patch,
        byte[] columns)
    {
        int rowFloats = outWidth * patch;

        Parallelism.For(rows, r =>
        {
            int outY = rowStart + r;
            Span<float> rowColumns = MemoryMarshal.Cast<byte, float>(
                columns.AsSpan(r * rowFloats * sizeof(float), rowFloats * sizeof(float)));

            for (int c = 0; c < channels; c++)
            {
                int channelBase = inputBase + (c * inHeight * inWidth);
                for (int ky = 0; ky < kernelHeight; ky++)
                {
                    int iy = (outY * strideY) - padTop + (ky * dilationY);
                    int tapBase = ((c * kernelHeight) + ky) * kernelWidth;
                    bool inRange = (uint)iy < (uint)inHeight;
                    int rowBase = channelBase + (iy * inWidth);

                    for (int kx = 0; kx < kernelWidth; kx++)
                    {
                        int offset = tapBase + kx;
                        int start = -padLeft + (kx * dilationX);

                        for (int ox = 0; ox < outWidth; ox++)
                        {
                            int ix = start + (ox * strideX);
                            rowColumns[(ox * patch) + offset] =
                                inRange && (uint)ix < (uint)inWidth ? input[rowBase + ix] : 0f;
                        }
                    }
                }
            }
        });
    }

    private static void Depthwise(
        PaddleTensor input,
        PaddleTensor weight,
        PaddleTensor result,
        int batch,
        int channels,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int outHeight,
        int outWidth,
        int[] strides,
        int[] dilations,
        int padTop,
        int padLeft)
    {
        float[] source = input.Floats!;
        float[] filters = weight.Floats!;
        float[] destination = result.Floats!;
        int kernelSize = kernelHeight * kernelWidth;
        int strideY = strides[0];
        int strideX = strides[1];
        int dilationY = dilations[0];
        int dilationX = dilations[1];

        // Output columns whose every horizontal tap lands inside the image. Only these can be
        // done a vector at a time; the few on either side keep the scalar path.
        int interiorStart = Math.Clamp(padLeft, 0, outWidth);
        int interiorEnd = Math.Clamp(inWidth - ((kernelWidth - 1) * dilationX) + padLeft, interiorStart, outWidth);
        bool vectorise = Simd.Use256 && strideX == 1 && interiorEnd - interiorStart >= Vector256<float>.Count;

        Parallelism.For(batch * channels, index =>
        {
            int n = index / channels;
            int c = index % channels;
            int inputBase = ((n * channels) + c) * inHeight * inWidth;
            int filterBase = c * kernelSize;
            int outputBase = ((n * channels) + c) * outHeight * outWidth;

            for (int oy = 0; oy < outHeight; oy++)
            {
                Span<float> row = destination.AsSpan(outputBase + (oy * outWidth), outWidth);
                int first = vectorise ? interiorStart : outWidth;
                int last = vectorise ? interiorEnd : outWidth;

                for (int ox = 0; ox < first; ox++)
                {
                    row[ox] = Tap(source, filters, inputBase, filterBase, inHeight, inWidth,
                        kernelHeight, kernelWidth, strideY, strideX, dilationY, dilationX,
                        padTop, padLeft, oy, ox);
                }

                if (vectorise)
                {
                    Interior(
                        source, filters, row, inputBase, filterBase, inHeight, inWidth,
                        kernelHeight, kernelWidth, strideY, dilationY, dilationX, padTop, padLeft,
                        oy, first, last);
                }

                for (int ox = last; ox < outWidth; ox++)
                {
                    row[ox] = Tap(source, filters, inputBase, filterBase, inHeight, inWidth,
                        kernelHeight, kernelWidth, strideY, strideX, dilationY, dilationX,
                        padTop, padLeft, oy, ox);
                }
            }
        });
    }

    /// <summary>One depthwise output, taps bounds-checked. The image border, and any stride.</summary>
    private static float Tap(
        float[] source,
        float[] filters,
        int inputBase,
        int filterBase,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int strideY,
        int strideX,
        int dilationY,
        int dilationX,
        int padTop,
        int padLeft,
        int outY,
        int outX)
    {
        float sum = 0f;

        for (int ky = 0; ky < kernelHeight; ky++)
        {
            int iy = (outY * strideY) - padTop + (ky * dilationY);
            if ((uint)iy >= (uint)inHeight)
            {
                continue;
            }

            int rowBase = inputBase + (iy * inWidth);
            int filterRow = filterBase + (ky * kernelWidth);

            for (int kx = 0; kx < kernelWidth; kx++)
            {
                int ix = (outX * strideX) - padLeft + (kx * dilationX);
                if ((uint)ix < (uint)inWidth)
                {
                    sum += source[rowBase + ix] * filters[filterRow + kx];
                }
            }
        }

        return sum;
    }

    /// <summary>
    /// The stride-1 interior of one depthwise output row, a vector of output columns at a time.
    /// </summary>
    /// <remarks>
    /// The accumulator stays in a register across the whole 5x5 window, so the row is stored once
    /// rather than once per tap. Written out per output column instead — which is what the
    /// operator did — a 128-channel 100x100 depthwise ran at 3.7 GFLOP/s, bounds-checking every
    /// one of its twenty-five taps.
    /// </remarks>
    private static void Interior(
        float[] source,
        float[] filters,
        Span<float> row,
        int inputBase,
        int filterBase,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int strideY,
        int dilationY,
        int dilationX,
        int padTop,
        int padLeft,
        int outY,
        int first,
        int last)
    {
        int width = Vector256<float>.Count;
        ref float input = ref MemoryMarshal.GetArrayDataReference(source);

        int ox = first;
        for (; ox <= last - width; ox += width)
        {
            Vector256<float> sum = Vector256<float>.Zero;

            for (int ky = 0; ky < kernelHeight; ky++)
            {
                int iy = (outY * strideY) - padTop + (ky * dilationY);
                if ((uint)iy >= (uint)inHeight)
                {
                    continue;
                }

                int rowBase = inputBase + (iy * inWidth) + ox - padLeft;
                int filterRow = filterBase + (ky * kernelWidth);

                for (int kx = 0; kx < kernelWidth; kx++)
                {
                    sum = Vector256.FusedMultiplyAdd(
                        Vector256.LoadUnsafe(ref input, (nuint)(rowBase + (kx * dilationX))),
                        Vector256.Create(filters[filterRow + kx]),
                        sum);
                }
            }

            sum.StoreUnsafe(ref MemoryMarshal.GetReference(row), (nuint)ox);
        }

        for (; ox < last; ox++)
        {
            float sum = 0f;

            for (int ky = 0; ky < kernelHeight; ky++)
            {
                int iy = (outY * strideY) - padTop + (ky * dilationY);
                if ((uint)iy >= (uint)inHeight)
                {
                    continue;
                }

                int rowBase = inputBase + (iy * inWidth) + ox - padLeft;
                int filterRow = filterBase + (ky * kernelWidth);

                for (int kx = 0; kx < kernelWidth; kx++)
                {
                    sum += source[rowBase + (kx * dilationX)] * filters[filterRow + kx];
                }
            }

            row[ox] = sum;
        }
    }

    /// <summary>2-D pooling, matching <c>pd_op.pool2d</c>.</summary>
    public static PaddleTensor Pool2d(
        PaddleTensor input,
        int[] kernel,
        int[] strides,
        int[] paddings,
        bool ceilMode,
        bool exclusive,
        bool globalPooling,
        bool adaptive,
        string poolingType,
        string paddingAlgorithm,
        string dataFormat)
    {
        if (dataFormat != "NCHW")
        {
            throw new NotSupportedException($"pool2d data format '{dataFormat}' is not supported.");
        }

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        if (globalPooling)
        {
            kernel = [inHeight, inWidth];
            paddings = [0, 0];
        }

        if (adaptive)
        {
            return Adaptive(input, kernel, poolingType);
        }

        (int padTop, int padBottom, int padLeft, int padRight) = ResolvePadding(
            paddingAlgorithm, paddings, inHeight, inWidth, kernel[0], kernel[1], strides, [1, 1]);

        int outHeight = OutputExtent(inHeight, kernel[0], strides[0], padTop, padBottom, ceilMode);
        int outWidth = OutputExtent(inWidth, kernel[1], strides[1], padLeft, padRight, ceilMode);

        PaddleTensor result = PaddleTensor.Float([batch, channels, outHeight, outWidth]);
        float[] source = input.Floats!;
        float[] destination = result.Floats!;
        bool isMax = poolingType == "max";

        Parallelism.For(batch * channels, index =>
        {
            int inputBase = index * inHeight * inWidth;
            int outputBase = index * outHeight * outWidth;

            for (int oy = 0; oy < outHeight; oy++)
            {
                int startY = (oy * strides[0]) - padTop;
                int endY = Math.Min(startY + kernel[0], inHeight);
                int clampedStartY = Math.Max(startY, 0);

                for (int ox = 0; ox < outWidth; ox++)
                {
                    int startX = (ox * strides[1]) - padLeft;
                    int endX = Math.Min(startX + kernel[1], inWidth);
                    int clampedStartX = Math.Max(startX, 0);

                    if (isMax)
                    {
                        float best = float.NegativeInfinity;
                        for (int y = clampedStartY; y < endY; y++)
                        {
                            int rowBase = inputBase + (y * inWidth);
                            for (int x = clampedStartX; x < endX; x++)
                            {
                                best = Math.Max(best, source[rowBase + x]);
                            }
                        }

                        destination[outputBase + (oy * outWidth) + ox] = best;
                    }
                    else
                    {
                        float sum = 0f;
                        for (int y = clampedStartY; y < endY; y++)
                        {
                            int rowBase = inputBase + (y * inWidth);
                            for (int x = clampedStartX; x < endX; x++)
                            {
                                sum += source[rowBase + x];
                            }
                        }

                        int area = exclusive
                            ? (endY - clampedStartY) * (endX - clampedStartX)
                            : kernel[0] * kernel[1];
                        destination[outputBase + (oy * outWidth) + ox] = area > 0 ? sum / area : 0f;
                    }
                }
            }
        });

        return result;
    }

    private static PaddleTensor Adaptive(PaddleTensor input, int[] output, string poolingType)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];
        int outHeight = output[0];
        int outWidth = output[1];

        PaddleTensor result = PaddleTensor.Float([batch, channels, outHeight, outWidth]);
        float[] source = input.Floats!;
        float[] destination = result.Floats!;
        bool isMax = poolingType == "max";

        Parallelism.For(batch * channels, index =>
        {
            int inputBase = index * inHeight * inWidth;
            int outputBase = index * outHeight * outWidth;

            for (int oy = 0; oy < outHeight; oy++)
            {
                int startY = oy * inHeight / outHeight;
                int endY = (((oy + 1) * inHeight) + outHeight - 1) / outHeight;

                for (int ox = 0; ox < outWidth; ox++)
                {
                    int startX = ox * inWidth / outWidth;
                    int endX = (((ox + 1) * inWidth) + outWidth - 1) / outWidth;

                    float accumulator = isMax ? float.NegativeInfinity : 0f;
                    for (int y = startY; y < endY; y++)
                    {
                        int rowBase = inputBase + (y * inWidth);
                        for (int x = startX; x < endX; x++)
                        {
                            accumulator = isMax
                                ? Math.Max(accumulator, source[rowBase + x])
                                : accumulator + source[rowBase + x];
                        }
                    }

                    destination[outputBase + (oy * outWidth) + ox] = isMax
                        ? accumulator
                        : accumulator / ((endY - startY) * (endX - startX));
                }
            }
        });

        return result;
    }

    private static int OutputExtent(int size, int kernel, int stride, int padBefore, int padAfter, bool ceilMode)
    {
        int numerator = size + padBefore + padAfter - kernel;
        int extent = ceilMode
            ? ((numerator + stride - 1) / stride) + 1
            : (numerator / stride) + 1;

        // Paddle drops a trailing window that starts inside the padding.
        if (ceilMode && (extent - 1) * stride >= size + padBefore)
        {
            extent--;
        }

        return Math.Max(extent, 1);
    }

    /// <summary>
    /// Resolves the four padding amounts, handling the <c>SAME</c> and <c>VALID</c> algorithms.
    /// </summary>
    private static (int Top, int Bottom, int Left, int Right) ResolvePadding(
        string algorithm,
        int[] paddings,
        int inHeight,
        int inWidth,
        int kernelHeight,
        int kernelWidth,
        int[] strides,
        int[] dilations)
    {
        switch (algorithm)
        {
            case "VALID":
                return (0, 0, 0, 0);

            case "SAME":
            {
                int neededHeight = Math.Max(
                    0,
                    ((((inHeight + strides[0] - 1) / strides[0]) - 1) * strides[0])
                        + (dilations[0] * (kernelHeight - 1)) + 1 - inHeight);
                int neededWidth = Math.Max(
                    0,
                    ((((inWidth + strides[1] - 1) / strides[1]) - 1) * strides[1])
                        + (dilations[1] * (kernelWidth - 1)) + 1 - inWidth);

                return (neededHeight / 2, neededHeight - (neededHeight / 2),
                    neededWidth / 2, neededWidth - (neededWidth / 2));
            }

            default:
                return paddings.Length switch
                {
                    2 => (paddings[0], paddings[0], paddings[1], paddings[1]),
                    4 => (paddings[0], paddings[1], paddings[2], paddings[3]),
                    1 => (paddings[0], paddings[0], paddings[0], paddings[0]),
                    _ => throw new NotSupportedException(
                        $"Unsupported padding array of length {paddings.Length}."),
                };
        }
    }
}
