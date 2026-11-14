using System.Diagnostics;
using PaddleOcrSharp.Download;

namespace PaddleOcrSharp.Cli;

/// <summary>Fetches models into the local cache.</summary>
public static class DownloadCommand
{
    /// <summary>Runs the <c>download</c> verb.</summary>
    public static async Task<int> RunAsync(CommandLine command)
    {
        List<ModelDescriptor> models = [];

        if (command.Positional.Count == 0)
        {
            models.Add(ModelCatalog.VisionLanguageFor(ModelLocator.QuantizationFrom(command)));
            models.Add(ModelCatalog.PpDocLayoutV3);
        }
        else
        {
            foreach (string name in command.Positional)
            {
                if (string.Equals(name, "all", StringComparison.OrdinalIgnoreCase))
                {
                    models.AddRange(ModelCatalog.All);
                    continue;
                }

                ModelDescriptor? model = ModelCatalog.Find(name);
                if (model is null)
                {
                    Console.Error.WriteLine(
                        $"Unknown model '{name}'. Known models: " +
                        string.Join(", ", ModelCatalog.All.Select(entry => entry.Name)) + ", all.");
                    return 2;
                }

                models.Add(model);
            }
        }

        using var downloader = new ModelDownloader(command.Get("cache"), command.Get("endpoint"));
        Console.WriteLine($"Cache: {downloader.CacheRoot}");
        Console.WriteLine($"Endpoint: {downloader.Endpoint}");

        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, eventArgs) =>
        {
            eventArgs.Cancel = true;
            cancellation.Cancel();
        };

        foreach (ModelDescriptor model in models.Distinct())
        {
            Console.WriteLine();
            Console.WriteLine($"{model.Name} ({model.RemotePath}/)");

            var reporter = new ConsoleProgress();
            try
            {
                string directory = await downloader
                    .EnsureAsync(model, reporter, cancellation.Token)
                    .ConfigureAwait(false);
                reporter.Finish();
                Console.WriteLine($"  ready: {directory}");
            }
            catch (OperationCanceledException)
            {
                Console.Error.WriteLine("  cancelled");
                return 130;
            }
            catch (Exception exception)
            {
                reporter.Finish();
                Console.Error.WriteLine($"  failed: {exception.Message}");
                return 1;
            }
        }

        return 0;
    }

    /// <summary>Prints a single refreshing progress line per file.</summary>
    private sealed class ConsoleProgress : IProgress<DownloadProgress>
    {
        private readonly Stopwatch _clock = Stopwatch.StartNew();
        private string _current = string.Empty;
        private long _lastRender;

        public void Report(DownloadProgress value)
        {
            if (value.File != _current)
            {
                Finish();
                _current = value.File;
                _lastRender = 0;
            }

            if (value.Cached)
            {
                Console.WriteLine($"  {value.File}: cached ({Format(value.BytesReceived)})");
                _current = string.Empty;
                return;
            }

            long elapsed = _clock.ElapsedMilliseconds;
            if (elapsed - _lastRender < 250 && value.Fraction is not >= 1)
            {
                return;
            }

            _lastRender = elapsed;
            string amount = value.TotalBytes is { } total
                ? $"{Format(value.BytesReceived)} / {Format(total)} ({value.Fraction * 100:F1}%)"
                : Format(value.BytesReceived);

            Console.Write($"\r  {value.File}: {amount}          ");
        }

        public void Finish()
        {
            if (_current.Length > 0)
            {
                Console.WriteLine();
                _current = string.Empty;
            }
        }

        private static string Format(long bytes) => bytes switch
        {
            >= 1L << 30 => $"{bytes / (double)(1L << 30):F2} GiB",
            >= 1L << 20 => $"{bytes / (double)(1L << 20):F1} MiB",
            >= 1L << 10 => $"{bytes / (double)(1L << 10):F1} KiB",
            _ => $"{bytes} B",
        };
    }
}
