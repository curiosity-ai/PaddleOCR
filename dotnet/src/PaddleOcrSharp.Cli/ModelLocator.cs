using PaddleOcrSharp.Download;
using PaddleOcrSharp.Models;

namespace PaddleOcrSharp.Cli;

/// <summary>Resolves a checkpoint directory from the command line, environment or cache.</summary>
public static class ModelLocator
{
    /// <summary>
    /// Finds a catalogued model in the cache, downloading it when <paramref name="allowDownload"/>
    /// is set and returning <see langword="null"/> when it is neither cached nor downloadable.
    /// </summary>
    public static async Task<string?> ResolveOptionalAsync(
        CommandLine command,
        ModelDescriptor model,
        string overrideOption,
        string environmentVariable,
        bool allowDownload,
        CancellationToken cancellationToken = default)
    {
        string? explicitPath = command.Get(overrideOption)
            ?? Environment.GetEnvironmentVariable(environmentVariable);

        if (!string.IsNullOrEmpty(explicitPath))
        {
            return explicitPath;
        }

        using var downloader = new ModelDownloader(command.Get("cache"), command.Get("endpoint"));
        if (downloader.IsComplete(model))
        {
            return downloader.DirectoryFor(model);
        }

        if (!allowDownload)
        {
            return null;
        }

        Console.Error.WriteLine($"Fetching {model.Name} into {downloader.CacheRoot} …");
        return await downloader.EnsureAsync(model, progress: null, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Reads how the language model's weights should be held: <c>--quantization none|int8</c>, or the
    /// <c>PADDLEOCR_SHARP_QUANTIZATION</c> environment variable, or none.
    /// </summary>
    public static WeightQuantization QuantizationFrom(CommandLine command)
    {
        string? text = command.Get("quantization") ?? Environment.GetEnvironmentVariable("PADDLEOCR_SHARP_QUANTIZATION");
        if (string.IsNullOrWhiteSpace(text) || text.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return WeightQuantization.None;
        }

        if (text.Equals("int8", StringComparison.OrdinalIgnoreCase))
        {
            return WeightQuantization.Int8;
        }

        throw new ArgumentException($"Unknown quantization '{text}'. Use none or int8.");
    }

    /// <summary>
    /// Finds the layout detector, downloading it when <paramref name="allowDownload"/> is set.
    /// </summary>
    public static async Task<string> ResolveLayoutAsync(
        CommandLine command,
        bool allowDownload,
        CancellationToken cancellationToken = default)
    {
        string? explicitPath = command.Get("layout-dir")
            ?? Environment.GetEnvironmentVariable("PP_DOCLAYOUT_V3_DIR");

        if (!string.IsNullOrEmpty(explicitPath))
        {
            if (!File.Exists(Path.Combine(explicitPath, "inference.json")))
            {
                throw new DirectoryNotFoundException($"'{explicitPath}' does not contain inference.json.");
            }

            return explicitPath;
        }

        using var downloader = new ModelDownloader(command.Get("cache"), command.Get("endpoint"));
        ModelDescriptor model = ModelCatalog.PpDocLayoutV3;

        if (downloader.IsComplete(model))
        {
            return downloader.DirectoryFor(model);
        }

        if (!allowDownload)
        {
            throw new FileNotFoundException(
                $"Model '{model.Name}' is not in the cache at {downloader.DirectoryFor(model)}. " +
                "Run `paddleocr-sharp download` first, or pass --layout-dir.");
        }

        Console.Error.WriteLine($"Fetching {model.Name} into {downloader.CacheRoot} …");
        return await downloader.EnsureAsync(model, progress: null, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Finds the VL checkpoint, downloading it when <paramref name="allowDownload"/> is set.
    /// </summary>
    public static async Task<string> ResolveVLAsync(
        CommandLine command,
        bool allowDownload,
        CancellationToken cancellationToken = default)
    {
        string? explicitPath = command.Get("model-dir")
            ?? Environment.GetEnvironmentVariable("PADDLEOCR_VL_DIR");

        if (!string.IsNullOrEmpty(explicitPath))
        {
            if (!File.Exists(Path.Combine(explicitPath, "model.safetensors")))
            {
                throw new DirectoryNotFoundException(
                    $"'{explicitPath}' does not contain model.safetensors.");
            }

            return explicitPath;
        }

        using var downloader = new ModelDownloader(command.Get("cache"), command.Get("endpoint"));
        ModelDescriptor model = ModelCatalog.PaddleOcrVL16;

        if (downloader.IsComplete(model))
        {
            return downloader.DirectoryFor(model);
        }

        if (!allowDownload)
        {
            throw new FileNotFoundException(
                $"Model '{model.Name}' is not in the cache at {downloader.DirectoryFor(model)}. " +
                "Run `paddleocr-sharp download` first, or pass --model-dir.");
        }

        Console.Error.WriteLine($"Fetching {model.Name} into {downloader.CacheRoot} …");
        return await downloader.EnsureAsync(model, progress: null, cancellationToken).ConfigureAwait(false);
    }
}
