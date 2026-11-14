using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Models.Paddle.Ops;

/// <summary>Resampling kernels: grid sampling and interpolation.</summary>
internal static class SamplingOps
{
    /// <summary>
    /// Bilinear grid sampling, matching <c>pd_op.grid_sample</c>.
    /// </summary>
    /// <remarks>
    /// This is the kernel behind deformable attention: the decoder samples the encoder feature
    /// maps at learned offsets. Only the combination the exported program uses is implemented —
    /// bilinear mode, zero padding, <c>align_corners=false</c> — and anything else throws rather
    /// than silently sampling differently.
    /// </remarks>
    /// <param name="input">Feature map, <c>[N, C, H, W]</c>.</param>
    /// <param name="grid">Sample locations in <c>[-1, 1]</c>, <c>[N, outH, outW, 2]</c> as (x, y).</param>
    /// <param name="mode">Interpolation mode; only <c>bilinear</c> and <c>nearest</c> are supported.</param>
    /// <param name="paddingMode">Out-of-range behaviour: <c>zeros</c>, <c>border</c> or <c>reflection</c>.</param>
    /// <param name="alignCorners">Whether −1 and 1 address pixel centres rather than corners.</param>
    public static PaddleTensor GridSample(
        PaddleTensor input,
        PaddleTensor grid,
        string mode,
        string paddingMode,
        bool alignCorners)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];
        int outHeight = grid.Shape[1];
        int outWidth = grid.Shape[2];

        PaddleTensor result = PaddleTensor.Float([batch, channels, outHeight, outWidth]);
        float[] source = input.Floats!;
        float[] coordinates = grid.Floats!;
        float[] destination = result.Floats!;

        bool bilinear = mode switch
        {
            "bilinear" => true,
            "nearest" => false,
            _ => throw new NotSupportedException($"grid_sample mode '{mode}' is not supported."),
        };

        Parallelism.For(batch * outHeight, index =>
        {
            int n = index / outHeight;
            int oy = index % outHeight;

            for (int ox = 0; ox < outWidth; ox++)
            {
                int gridOffset = ((((n * outHeight) + oy) * outWidth) + ox) * 2;
                float gx = coordinates[gridOffset];
                float gy = coordinates[gridOffset + 1];

                float x = Unnormalise(gx, inWidth, alignCorners);
                float y = Unnormalise(gy, inHeight, alignCorners);

                if (bilinear)
                {
                    int x0 = (int)MathF.Floor(x);
                    int y0 = (int)MathF.Floor(y);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    float wx1 = x - x0;
                    float wx0 = 1f - wx1;
                    float wy1 = y - y0;
                    float wy0 = 1f - wy1;

                    for (int c = 0; c < channels; c++)
                    {
                        int planeBase = ((n * channels) + c) * inHeight * inWidth;
                        float value =
                            (Sample(source, planeBase, inHeight, inWidth, x0, y0, paddingMode) * wx0 * wy0) +
                            (Sample(source, planeBase, inHeight, inWidth, x1, y0, paddingMode) * wx1 * wy0) +
                            (Sample(source, planeBase, inHeight, inWidth, x0, y1, paddingMode) * wx0 * wy1) +
                            (Sample(source, planeBase, inHeight, inWidth, x1, y1, paddingMode) * wx1 * wy1);

                        destination[((((n * channels) + c) * outHeight + oy) * outWidth) + ox] = value;
                    }
                }
                else
                {
                    int nx = (int)MathF.Round(x, MidpointRounding.AwayFromZero);
                    int ny = (int)MathF.Round(y, MidpointRounding.AwayFromZero);

                    for (int c = 0; c < channels; c++)
                    {
                        int planeBase = ((n * channels) + c) * inHeight * inWidth;
                        destination[((((n * channels) + c) * outHeight + oy) * outWidth) + ox] =
                            Sample(source, planeBase, inHeight, inWidth, nx, ny, paddingMode);
                    }
                }
            }
        });

        return result;
    }

    private static float Unnormalise(float coordinate, int size, bool alignCorners) =>
        alignCorners
            ? ((coordinate + 1f) / 2f) * (size - 1)
            : (((coordinate + 1f) * size) - 1f) / 2f;

    private static float Sample(
        float[] data,
        int planeBase,
        int height,
        int width,
        int x,
        int y,
        string paddingMode)
    {
        switch (paddingMode)
        {
            case "border":
                x = Math.Clamp(x, 0, width - 1);
                y = Math.Clamp(y, 0, height - 1);
                break;

            case "reflection":
                x = Reflect(x, width);
                y = Reflect(y, height);
                break;

            default:
                if ((uint)x >= (uint)width || (uint)y >= (uint)height)
                {
                    return 0f;
                }

                break;
        }

        return data[planeBase + (y * width) + x];
    }

    private static int Reflect(int index, int size)
    {
        if (size <= 1)
        {
            return 0;
        }

        int period = 2 * (size - 1);
        index = Math.Abs(index) % period;
        return index < size ? index : period - index;
    }

    /// <summary>Bilinear resize, matching <c>pd_op.bilinear_interp</c>.</summary>
    public static PaddleTensor BilinearInterp(
        PaddleTensor input,
        int outHeight,
        int outWidth,
        bool alignCorners,
        int alignMode,
        string dataFormat)
    {
        if (dataFormat != "NCHW")
        {
            throw new NotSupportedException($"bilinear_interp data format '{dataFormat}' is not supported.");
        }

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        PaddleTensor result = PaddleTensor.Float([batch, channels, outHeight, outWidth]);
        float[] source = input.Floats!;
        float[] destination = result.Floats!;

        float scaleY = alignCorners && outHeight > 1
            ? (float)(inHeight - 1) / (outHeight - 1)
            : (float)inHeight / outHeight;
        float scaleX = alignCorners && outWidth > 1
            ? (float)(inWidth - 1) / (outWidth - 1)
            : (float)inWidth / outWidth;

        Parallelism.For(batch * channels, plane =>
        {
            int inputBase = plane * inHeight * inWidth;
            int outputBase = plane * outHeight * outWidth;

            for (int oy = 0; oy < outHeight; oy++)
            {
                float sourceY = SourceIndex(oy, scaleY, alignCorners, alignMode);
                int y0 = (int)sourceY;
                int y1 = Math.Min(y0 + 1, inHeight - 1);
                float wy1 = sourceY - y0;
                float wy0 = 1f - wy1;

                for (int ox = 0; ox < outWidth; ox++)
                {
                    float sourceX = SourceIndex(ox, scaleX, alignCorners, alignMode);
                    int x0 = (int)sourceX;
                    int x1 = Math.Min(x0 + 1, inWidth - 1);
                    float wx1 = sourceX - x0;
                    float wx0 = 1f - wx1;

                    destination[outputBase + (oy * outWidth) + ox] =
                        (source[inputBase + (y0 * inWidth) + x0] * wy0 * wx0) +
                        (source[inputBase + (y0 * inWidth) + x1] * wy0 * wx1) +
                        (source[inputBase + (y1 * inWidth) + x0] * wy1 * wx0) +
                        (source[inputBase + (y1 * inWidth) + x1] * wy1 * wx1);
                }
            }
        });

        return result;
    }

    private static float SourceIndex(int index, float scale, bool alignCorners, int alignMode)
    {
        if (alignCorners)
        {
            return index * scale;
        }

        // align_mode 0 offsets by half a pixel, which is what torch/PIL call align_corners=False.
        float value = alignMode == 0 ? ((index + 0.5f) * scale) - 0.5f : index * scale;
        return value < 0f ? 0f : value;
    }

    /// <summary>Nearest-neighbour resize, matching <c>pd_op.nearest_interp</c>.</summary>
    public static PaddleTensor NearestInterp(
        PaddleTensor input,
        int outHeight,
        int outWidth,
        bool alignCorners,
        string dataFormat)
    {
        if (dataFormat != "NCHW")
        {
            throw new NotSupportedException($"nearest_interp data format '{dataFormat}' is not supported.");
        }

        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        PaddleTensor result = PaddleTensor.Float([batch, channels, outHeight, outWidth]);
        float[] source = input.Floats!;
        float[] destination = result.Floats!;

        float scaleY = alignCorners && outHeight > 1
            ? (float)(inHeight - 1) / (outHeight - 1)
            : (float)inHeight / outHeight;
        float scaleX = alignCorners && outWidth > 1
            ? (float)(inWidth - 1) / (outWidth - 1)
            : (float)inWidth / outWidth;

        Parallelism.For(batch * channels, plane =>
        {
            int inputBase = plane * inHeight * inWidth;
            int outputBase = plane * outHeight * outWidth;

            for (int oy = 0; oy < outHeight; oy++)
            {
                int sy = alignCorners
                    ? (int)MathF.Round(oy * scaleY, MidpointRounding.AwayFromZero)
                    : (int)(oy * scaleY);
                sy = Math.Min(sy, inHeight - 1);

                for (int ox = 0; ox < outWidth; ox++)
                {
                    int sx = alignCorners
                        ? (int)MathF.Round(ox * scaleX, MidpointRounding.AwayFromZero)
                        : (int)(ox * scaleX);
                    sx = Math.Min(sx, inWidth - 1);

                    destination[outputBase + (oy * outWidth) + ox] = source[inputBase + (sy * inWidth) + sx];
                }
            }
        });

        return result;
    }
}
