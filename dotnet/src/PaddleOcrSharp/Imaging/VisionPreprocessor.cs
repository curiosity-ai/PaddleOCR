using PaddleOcrSharp.Core;

namespace PaddleOcrSharp.Imaging;

/// <summary>Grid shape of one preprocessed image, in patches.</summary>
/// <param name="Temporal">Temporal extent; always 1 for still images.</param>
/// <param name="Height">Patch rows.</param>
/// <param name="Width">Patch columns.</param>
public readonly record struct ImageGrid(int Temporal, int Height, int Width)
{
    /// <summary>Total number of patches.</summary>
    public int PatchCount => Temporal * Height * Width;

    /// <summary>Number of language-model tokens the image expands to after 2×2 merging.</summary>
    public int TokenCount(int mergeSize) => PatchCount / (mergeSize * mergeSize);
}

/// <summary>One image, patchified and normalised, ready for the vision tower.</summary>
public sealed class PreprocessedImage : IDisposable
{
    /// <summary>Wraps already-patchified pixel values.</summary>
    /// <param name="pixelValues">Patches as <c>[patchCount, channels × patchSize × patchSize]</c>.</param>
    /// <param name="grid">Patch grid the values describe.</param>
    public PreprocessedImage(Tensor pixelValues, ImageGrid grid)
    {
        PixelValues = pixelValues;
        Grid = grid;
    }

    /// <summary>Patches as <c>[patchCount, channels × patchSize × patchSize]</c>.</summary>
    public Tensor PixelValues { get; }

    /// <summary>Patch grid of the image.</summary>
    public ImageGrid Grid { get; }

    /// <inheritdoc />
    public void Dispose() => PixelValues.Dispose();
}

/// <summary>
/// Port of <c>PaddleOCRVLImageProcessor._preprocess</c>: smart-resize, bicubic resample,
/// rescale, normalise, then flatten to per-patch vectors.
/// </summary>
public static class VisionPreprocessor
{
    /// <summary>Preprocesses <paramref name="image"/> with the supplied options.</summary>
    public static PreprocessedImage Preprocess(RgbImage image, VisionPreprocessorOptions options)
    {
        (int targetHeight, int targetWidth) = SmartResize.Compute(
            image.Height, image.Width, options.Factor, options.MinPixels, options.MaxPixels);

        RgbImage resized = PilResize.ResizeBicubic(image, targetWidth, targetHeight);
        try
        {
            return Patchify(resized, options);
        }
        finally
        {
            if (!ReferenceEquals(resized, image))
            {
                resized.Dispose();
            }
        }
    }

    /// <summary>
    /// Converts an already-resized image into normalised per-patch vectors.
    /// </summary>
    /// <remarks>
    /// Upstream reshapes <c>[t, tps, C, gh, p, gw, p]</c> and permutes to
    /// <c>[t, gh, gw, C, tps, p, p]</c>, so patch <c>(row, col)</c> lands at index
    /// <c>row · gridWidth + col</c> with its channels outermost. That is what this writes.
    /// </remarks>
    public static PreprocessedImage Patchify(RgbImage image, VisionPreprocessorOptions options)
    {
        int patch = options.PatchSize;
        if (image.Width % patch != 0 || image.Height % patch != 0)
        {
            throw new ArgumentException(
                $"A {image.Width}×{image.Height} image is not a whole number of {patch}px patches.",
                nameof(image));
        }

        int gridWidth = image.Width / patch;
        int gridHeight = image.Height / patch;
        var grid = new ImageGrid(options.TemporalPatchSize, gridHeight, gridWidth);

        int patchLength = 3 * patch * patch;
        Tensor pixelValues = Tensor.Rent(grid.PatchCount, patchLength);

        double rescale = options.RescaleFactor;
        float[] mean = options.ImageMean;
        float[] std = options.ImageStd;

        Memory<float> destination = pixelValues.Memory;

        Parallelism.For(gridHeight, gy =>
        {
            Span<float> output = destination.Span;
            for (int gx = 0; gx < gridWidth; gx++)
            {
                Span<float> target = output.Slice((((gy * gridWidth) + gx) * patchLength), patchLength);

                for (int c = 0; c < 3; c++)
                {
                    float channelMean = mean[c];
                    float channelStd = std[c];
                    int channelBase = c * patch * patch;

                    for (int py = 0; py < patch; py++)
                    {
                        ReadOnlySpan<byte> row = image.Row((gy * patch) + py);
                        int rowBase = (gx * patch * 3) + c;
                        int outBase = channelBase + (py * patch);

                        for (int px = 0; px < patch; px++)
                        {
                            // Upstream rescales in float64 (`uint8 * (1/255)`) and then narrows to
                            // float32 before normalising, so mirror both steps.
                            float value = (float)(row[rowBase + (px * 3)] * rescale);
                            target[outBase + px] = (value - channelMean) / channelStd;
                        }
                    }
                }
            }
        });

        return new PreprocessedImage(pixelValues, grid);
    }
}
