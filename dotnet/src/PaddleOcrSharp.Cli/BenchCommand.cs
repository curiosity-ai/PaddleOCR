using System.Diagnostics;
using PaddleOcrSharp.Core;
using PaddleOcrSharp.Imaging;
using PaddleOcrSharp.Models;
using PaddleOcrSharp.Models.Vision;
using PaddleOcrSharp.Models.Layout;
using PaddleOcrSharp.Models.Paddle;

namespace PaddleOcrSharp.Cli;

/// <summary>Measures per-stage throughput and allocation behaviour.</summary>
public static class BenchCommand
{
    /// <summary>Runs the <c>bench</c> verb.</summary>
    public static async Task<int> RunAsync(CommandLine command)
    {
        int width = command.GetInt("width", 1024);
        int height = command.GetInt("height", 1024);
        int iterations = Math.Max(1, command.GetInt("iterations", 3));
        bool benchmarkVL = command.GetBool("vl", true);

        // Before anything is loaded, and before the model work has had a chance to warm or
        // pollute anything, find out what this machine will actually do today. Everything below
        // is reported against these ceilings as well as in milliseconds.
        MachineProfile? machine = command.GetBool("calibrate", true)
            ? MachineProfile.Measure(command.GetFloat("calibrate-seconds", 1.6f))
            : null;

        if (machine is not null)
        {
            Console.WriteLine(machine.Describe());
            Console.WriteLine();
        }

        if (command.GetBool("gemm", false))
        {
            GemmBenchmark.Run(machine);
        }

        // A synthetic page is fine for the tower, which does the same work whatever the pixels
        // are, but not for the decoder: noise makes the model emit the stop token at once, so the
        // decode loop never runs. `--image` feeds a real crop instead.
        string? imagePath = command.Get("image");
        using RgbImage source = imagePath is null
            ? Synthetic(width, height)
            : ImageIO.Load(imagePath);

        if (benchmarkVL)
        {
            await BenchmarkVLAsync(command, source, iterations, machine).ConfigureAwait(false);
        }

        if (command.GetBool("layout", true))
        {
            await BenchmarkLayoutAsync(command, source, iterations).ConfigureAwait(false);
        }

        return 0;
    }

    /// <summary>Times the vision tower and the decoder.</summary>
    private static async Task BenchmarkVLAsync(
        CommandLine command, RgbImage source, int iterations, MachineProfile? machine)
    {
        string directory = await ModelLocator
            .ResolveVLAsync(command, allowDownload: true)
            .ConfigureAwait(false);

        var clock = Stopwatch.StartNew();
        using PaddleOcrVLModel model = PaddleOcrVLModel.Load(directory, ModelLocator.QuantizationFrom(command));
        Console.WriteLine($"Load: {clock.Elapsed.TotalSeconds:F2}s");

        clock.Restart();
        using PreprocessedImage preprocessed = VisionPreprocessor.Preprocess(
            source, VisionPreprocessorOptions.Default);
        Console.WriteLine(
            $"Preprocess: {clock.Elapsed.TotalMilliseconds:F0}ms " +
            $"-> grid {preprocessed.Grid.Height}x{preprocessed.Grid.Width} " +
            $"({preprocessed.Grid.PatchCount} patches)");

        for (int i = 0; i < iterations; i++)
        {
            var stages = i == iterations - 1 ? new StageProfile() : null;
            long before = GC.GetTotalAllocatedBytes(precise: true);
            clock.Restart();
            using Tensor embeddings = model.Vision.Encode(preprocessed, profile: stages);
            TimeSpan elapsed = clock.Elapsed;
            long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;

            Console.WriteLine(
                $"Vision  [{i}]: {elapsed.TotalMilliseconds:F0}ms " +
                $"({preprocessed.Grid.PatchCount / elapsed.TotalSeconds:F0} patches/s, " +
                $"{allocated / (1024.0 * 1024.0):F1} MiB allocated)");

            if (stages is not null)
            {
                Console.WriteLine();
                Console.WriteLine(stages);
                Console.WriteLine();
            }
        }

        int tokens = Math.Max(1, command.GetInt("tokens", 32));
        int[] prompt = model.BuildPrompt(preprocessed.Grid, "OCR:");
        using Tensor imageEmbeddings = model.Vision.Encode(preprocessed);

        for (int i = 0; i < iterations; i++)
        {
            var stages = i == iterations - 1 ? new StageProfile() : null;
            long before = GC.GetTotalAllocatedBytes(precise: true);
            clock.Restart();
            List<int> generated = model.Generate(
                prompt,
                imageEmbeddings,
                preprocessed.Grid,
                GenerationOptions.Default with { MaxNewTokens = tokens },
                out _,
                stages);
            TimeSpan elapsed = clock.Elapsed;
            long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;

            Console.WriteLine(
                $"Decode  [{i}]: {elapsed.TotalMilliseconds:F0}ms for prefill({prompt.Length}) + " +
                $"{generated.Count} tokens ({allocated / (1024.0 * 1024.0):F1} MiB allocated)");

            if (stages is not null)
            {
                Console.WriteLine();
                Console.WriteLine(stages);
                Console.WriteLine();
            }
        }

    }

    /// <summary>Times the layout graph and reports which operators dominate it.</summary>
    private static async Task BenchmarkLayoutAsync(CommandLine command, RgbImage page, int iterations)
    {
        string directory;
        try
        {
            directory = await ModelLocator.ResolveLayoutAsync(command, allowDownload: false).ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"Skipping layout benchmark: {exception.Message}");
            return;
        }

        using LayoutDetector detector = LayoutDetector.Load(directory);
        var clock = Stopwatch.StartNew();

        for (int i = 0; i < iterations; i++)
        {
            var profile = new PirProfile();
            clock.Restart();

            // Reported per iteration, because the interesting number is the second one: the graph
            // runs the same shapes every time, so once its buffers are pooled a later detection
            // should allocate almost nothing where the first still fills the pool.
            long allocated = GC.GetAllocatedBytesForCurrentThread();
            IReadOnlyList<LayoutBox> boxes = detector.Detect(page, LayoutOptions.Default, profile);
            allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;

            Console.WriteLine(
                $"Layout  [{i}]: {clock.Elapsed.TotalMilliseconds:F0}ms, {boxes.Count} regions, "
                + $"{allocated / (double)(1L << 20):F0} MiB allocated");

            if (i == iterations - 1)
            {
                Console.WriteLine();
                Console.WriteLine(profile.Report());
            }
        }
    }

    private static RgbImage Synthetic(int width, int height)
    {
        RgbImage image = RgbImage.Rent(width, height);
        var random = new Random(7);
        for (int y = 0; y < height; y++)
        {
            Span<byte> row = image.Row(y);
            for (int x = 0; x < width; x++)
            {
                byte value = (byte)(((x * 7) + (y * 13) + random.Next(0, 32)) & 0xFF);
                row[(x * 3) + 0] = value;
                row[(x * 3) + 1] = (byte)(255 - value);
                row[(x * 3) + 2] = (byte)((value * 3) & 0xFF);
            }
        }

        return image;
    }
}
